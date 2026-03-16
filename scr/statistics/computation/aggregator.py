from typing import Callable, Mapping, Iterable, Any
import numpy as np

from scr.config.numerics import WP
from scr.utils.filesystem import is_empty


def _apply_wp(result):
    """
    Recursively apply WP to scalar outputs.
    Supports:
        - scalar
        - dict[str, scalar]
        - Iterable(scalar)
    """
    if isinstance(result, Mapping):
        return {k: WP(v) for k, v in result.items()}
    elif isinstance(result, Iterable) and not isinstance(result, (str, bytes)):
        return type(result)(WP(v) for v in result)
    else:
        return WP(result)


def _dict_listify(
        per_object: list[dict[str, float]],
        global_stats: Mapping[str, float],
) -> dict[str, list[float]]:
    """
    Convert list[dict] → dict[list] using schema from global_stats.
    """

    keys = global_stats.keys()

    if is_empty(per_object):
        return {k: [] for k in keys}

    return {
        k: [obj[k] for obj in per_object]
        for k in keys
    }


def aggregate_kernel(
        *,
        kernel: Callable,
        objects: list[np.ndarray] | None = None,
        total_object: np.ndarray | None = None,
        listify: bool = False,
        reduce_global: Callable[[Any], Any] | None = None,
        **kernel_kwargs,
) -> dict:
    """
    Generic aggregation kernel.

    kernel return types supported:
        scalar
        dict[str, scalar]
    """

    per_object = [
        _apply_wp(kernel(obj, **kernel_kwargs))
        for obj in objects
    ] if objects is not None else []

    if reduce_global is not None:
        global_stats = _apply_wp(reduce_global(per_object))
    else:
        global_stats = _apply_wp(kernel(total_object, **kernel_kwargs))

    if listify and isinstance(global_stats, Mapping):
        per_object = _dict_listify(per_object, global_stats)

    return {"per_object": per_object, "global": global_stats}
