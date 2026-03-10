import numpy as np
import numbers
from collections import defaultdict
from typing import Sequence, Mapping

from scr.utils.collections import nested_defaultdict


def nested_cast_arrays_dtype(
        obj,
        dtype: np.dtype,
):
    """
    Recursively cast all NumPy arrays in a nested structure to a given dtype.

    Parameters
    ----------
    obj
        Arbitrarily nested structure (dict, defaultdict, list, tuple, etc.)
        containing NumPy arrays.
    dtype
        Target NumPy dtype (e.g. np.float32).

    Returns
    -------
    obj_casted
        Structure identical to `obj`, but with all NumPy arrays cast to `dtype`.

    Notes
    -----
    - Container types are preserved (dict, defaultdict, list, tuple)
    - defaultdict default_factory is preserved
    - Only NumPy arrays are cast
    - Non-array objects are returned unchanged
    """

    # --------------------------------------------------
    # NumPy arrays
    # --------------------------------------------------
    if isinstance(obj, np.ndarray):
        if obj.dtype == dtype:
            return obj
        return obj.astype(dtype, copy=False)

    # --------------------------------------------------
    # defaultdict (preserve factory)
    # --------------------------------------------------
    if isinstance(obj, defaultdict):
        out = nested_defaultdict(factory=obj.default_factory)
        for k, v in obj.items():
            out[k] = nested_cast_arrays_dtype(v, dtype)
        return out

    # --------------------------------------------------
    # dict
    # --------------------------------------------------
    if isinstance(obj, dict):
        return {
            k: nested_cast_arrays_dtype(v, dtype)
            for k, v in obj.items()
        }

    # --------------------------------------------------
    # list / tuple
    # --------------------------------------------------
    if isinstance(obj, list):
        return [nested_cast_arrays_dtype(v, dtype) for v in obj]

    if isinstance(obj, tuple):
        return tuple(nested_cast_arrays_dtype(v, dtype) for v in obj)

    # --------------------------------------------------
    # everything else (scalars, None, etc.)
    # --------------------------------------------------
    return obj


def nested_equal(
        a,
        b,
        *,
        atol: float = 0.0,
        verbose: bool | int = False
) -> bool:
    """
    Compare two nested structures for equality.

    Rules
    -----
    - NaN == NaN is True
    - Numeric values use absolute tolerance `atol`
    - Supports dicts, lists, tuples, numpy arrays, and scalars
    - Early-exit on first difference if verbose=False

    Parameters
    ----------
    a, b
        Objects to compare.
    atol : float, keyword-only
        Maximum allowed absolute difference for numeric values.
    verbose : bool or int, keyword-only
        False  -> fast boolean check, early stop
        True   -> print all differences
        int N  -> print at most N differences

    Returns
    -------
    bool
        True if equal under the given rules, False otherwise.
    """

    early_exit = not verbose

    if verbose is True:
        max_reports = None
    elif isinstance(verbose, int) and verbose > 0:
        max_reports = verbose
    else:
        max_reports = 0

    reports: list[tuple[tuple, object, object, float | None]] = []

    def _add_report(path, x, y, diff):
        if max_reports == 0:
            return
        if max_reports is None or len(reports) < max_reports:
            reports.append((path, x, y, diff))

    def _compare(x, y, path) -> bool:
        # --------------------------------------------------
        # Float scalars
        # --------------------------------------------------
        if isinstance(x, (float, np.floating)) and isinstance(y, (float, np.floating)):
            if np.isnan(x) and np.isnan(y):
                return True

            diff = abs(x - y)
            if diff > atol:
                _add_report(path, x, y, diff)
                return False
            return True

        # --------------------------------------------------
        # Type mismatch
        # --------------------------------------------------
        if type(x) is not type(y):
            _add_report(path, x, y, None)
            return False

        # --------------------------------------------------
        # Dictionaries
        # --------------------------------------------------
        if isinstance(x, Mapping):
            equal = True
            keys = set(x) | set(y)
            for k in keys:
                if k not in x or k not in y:
                    _add_report(path + (k,), x.get(k), y.get(k), None)
                    if early_exit:
                        return False
                    equal = False
                else:
                    if not _compare(x[k], y[k], path + (k,)):
                        if early_exit:
                            return False
                        equal = False
            return equal

        # --------------------------------------------------
        # NumPy arrays
        # --------------------------------------------------
        if isinstance(x, np.ndarray):
            if x.shape != y.shape:
                _add_report(path, x.shape, y.shape, None)
                return False

            mask = ~np.isclose(x, y, atol=atol, rtol=0.0, equal_nan=True)
            if np.any(mask):
                equal = True
                for idx in map(tuple, np.argwhere(mask)):
                    diff = abs(x[idx] - y[idx])
                    _add_report(path + idx, x[idx], y[idx], diff)
                    if early_exit:
                        return False
                    equal = False
                return equal
            return True

        # --------------------------------------------------
        # Sequences (lists / tuples)
        # --------------------------------------------------
        if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
            if len(x) != len(y):
                _add_report(path, len(x), len(y), None)
                return False

            equal = True
            for i, (xx, yy) in enumerate(zip(x, y)):
                if not _compare(xx, yy, path + (i,)):
                    if early_exit:
                        return False
                    equal = False
            return equal

        # --------------------------------------------------
        # Fallback (exact comparison)
        # --------------------------------------------------
        if x != y:
            _add_report(path, x, y, None)
            return False

        return True

    equal = _compare(a, b, path=())

    # ------------------------------------------------------
    # Verbose output
    # ------------------------------------------------------
    if verbose and reports:
        for path, x, y, diff in reports:
            loc = ".".join(map(str, path)) if path else "<root>"
            if diff is None:
                print(f"Difference at {loc}: {x!r} != {y!r}")
            else:
                print(
                    f"Difference at {loc}: "
                    f"{x!r} vs {y!r} (|Δ|={diff:.3e} > atol={atol:.3e})"
                )

    return equal


def nested_cast(
        a,
        b
):
    """
    Recursively cast `b` to match the structure and types of `a`.

    Key guarantees
    --------------
    - defaultdict is preserved WITH default_factory
    - dict ↔ defaultdict allowed
    - exact structure required
    - safe numeric casts only
    """

    def _cast(x, y, path):
        # --------------------------------------------------
        # NumPy arrays
        # --------------------------------------------------
        if isinstance(x, np.ndarray):
            if not isinstance(y, np.ndarray):
                raise TypeError(f"{path}: expected ndarray, got {type(y).__name__}")

            if x.shape != y.shape:
                raise ValueError(f"{path}: shape mismatch {x.shape} vs {y.shape}")

            if np.issubdtype(x.dtype, np.integer) and np.issubdtype(y.dtype, np.floating):
                raise TypeError(f"{path}: refusing float → int array cast")

            return y.astype(x.dtype, copy=False)

        # --------------------------------------------------
        # defaultdict (PRESERVE FACTORY)
        # --------------------------------------------------
        if isinstance(x, defaultdict):
            if not isinstance(y, Mapping):
                raise TypeError(f"{path}: expected mapping, got {type(y).__name__}")

            if set(x) != set(y):
                diff = set(x) ^ set(y)
                raise ValueError(f"{path}: key mismatch {diff}")

            out = nested_defaultdict(factory=x.default_factory)
            for k in x:
                out[k] = _cast(x[k], y[k], f"{path}.{k}")
            return out

        # --------------------------------------------------
        # dict
        # --------------------------------------------------
        elif isinstance(x, dict):
            if not isinstance(y, Mapping):
                raise TypeError(f"{path}: expected mapping, got {type(y).__name__}")

            if set(x) != set(y):
                diff = set(x) ^ set(y)
                raise ValueError(f"{path}: key mismatch {diff}")

            return {
                k: _cast(x[k], y[k], f"{path}.{k}")
                for k in x
            }

        # --------------------------------------------------
        # Sequences
        # --------------------------------------------------
        if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
            if not isinstance(y, Sequence) or isinstance(y, (str, bytes)):
                raise TypeError(f"{path}: expected sequence, got {type(y).__name__}")

            if len(x) != len(y):
                raise ValueError(f"{path}: length mismatch {len(x)} vs {len(y)}")

            return type(x)(
                _cast(xx, yy, f"{path}[{i}]")
                for i, (xx, yy) in enumerate(zip(x, y))
            )

        # --------------------------------------------------
        # None
        # --------------------------------------------------
        if x is None:
            if y is not None:
                raise TypeError(f"{path}: cannot cast non-None to None")
            return None

        # --------------------------------------------------
        # Numeric scalars
        # --------------------------------------------------
        if isinstance(x, numbers.Number) and isinstance(y, numbers.Number):
            if isinstance(x, numbers.Real) and isinstance(y, numbers.Complex) and not isinstance(y, numbers.Real):
                raise TypeError(f"{path}: refusing complex → real cast")

            if isinstance(x, numbers.Integral) and isinstance(y, numbers.Real) and not isinstance(y, numbers.Integral):
                raise TypeError(f"{path}: refusing float → int cast")

            return type(x)(y)

        # --------------------------------------------------
        # Exact type fallback
        # --------------------------------------------------
        if type(x) is not type(y):
            raise TypeError(f"{path}: type mismatch {type(x).__name__} vs {type(y).__name__}")

        return y

    return _cast(a, b, "<root>")
