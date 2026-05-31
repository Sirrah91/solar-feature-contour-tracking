import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from typing import Callable

from scr.utils.types_alias import FrameFilteringMode
from scr.utils.filesystem import is_empty


def filter_dataset_by_spots(
        dataset: ds.FileSystemDataset,
        filtering_kwargs: dict,
        batch_size: int = 20000
) -> pd.DataFrame:
    """
    Filter a parquet dataset while preserving spot_global_index groups.

    Args:
        dataset: pyarrow dataset
        filtering_kwargs: same dict as filter_combined_df
        batch_size: number of rows per batch (Arrow batch size)

    Returns:
        filtered pandas DataFrame
    """

    dfs = []
    carry_over = pd.DataFrame()

    scanner = dataset.scanner(batch_size=batch_size)

    for batch in scanner.to_batches():
        df = batch.to_pandas()

        # prepend leftover from previous batch
        if not carry_over.empty:
            df = pd.concat([carry_over, df], ignore_index=True)
            carry_over = pd.DataFrame()

        if df.empty:
            continue

        # identify last group (potentially incomplete)
        last_key = df["spot_global_index"].iloc[-1]
        mask_last = df["spot_global_index"] == last_key

        complete_df = df[~mask_last]
        carry_over = df[mask_last]

        # safe: only full groups
        if not complete_df.empty:
            filtered = filter_combined_df(complete_df, filtering_kwargs)
            if not filtered.empty:
                dfs.append(filtered)

    # process final carry_over
    if not carry_over.empty:
        filtered = filter_combined_df(carry_over, filtering_kwargs)
        if not filtered.empty:
            dfs.append(filtered)

    # concat all filtered groups
    final_df = pd.concat(dfs, ignore_index=True)

    return final_df


def filter_combined_df(
        df: pd.DataFrame,
        filtering_kwargs: dict
) -> pd.DataFrame:
    """
    Filter a combined sunspot statistics DataFrame using flexible, hierarchical criteria.
    """

    group_cols = ["observation_id", "sunspot_id"]
    if set(group_cols).issubset(df.columns):
        # MUST BE PRESENT FOR `MODE IN ["ALL", "ANY"]`, NOT FOR `MODE == "FRAME-WISE"
        # Precompute group keys ONCE (important for speed & correctness)
        group_keys = df[group_cols].apply(tuple, axis=1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_column_name(part: str, param: str, stats_key: str | None = None) -> str:
        if "flux" in param or "variation" in param:
            if stats_key is None:
                raise ValueError(f"'stats_key' required for flux parameter '{param}'")
            return f"{stats_key}_{part}_{param}"

        return f"{part}_{param}"

    def _cell_satisfies(
            value,
            min_val: float | None = None,
            max_val: float | None = None,
            exact_val: float | str | None = None,
    ) -> bool:
        """
        Scalar or array-like test.
        - Scalars → treated as length-1 arrays
        - Arrays → ALL elements must satisfy
        - NaN / empty → False
        """
        if value is None:
            return False

        if isinstance(value, float) and np.isnan(value):
            return False

        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value)
            if is_empty(arr):
                return False
        else:
            arr = np.asarray([value])

        if exact_val is not None:
            return bool(np.all(arr == exact_val))

        cond = np.ones(arr.shape, dtype=bool)

        if min_val is not None:
            cond &= arr >= min_val
        if max_val is not None:
            cond &= arr <= max_val

        return bool(np.all(cond))

    def _scalar_satisfies(
            arr: np.ndarray,
            min_val: float | None = None,
            max_val: float | None = None,
            exact_val: float | str | None = None,
    ) -> np.ndarray:
        """Vectorised scalar-only version."""
        mask = np.isfinite(arr)

        if exact_val is not None:
            return mask & (arr == exact_val)

        if min_val is not None:
            mask &= arr >= min_val
        if max_val is not None:
            mask &= arr <= max_val

        return mask

    # ------------------------------------------------------------------
    # Core filtering logic
    # ------------------------------------------------------------------

    def _apply_filter(
            df: pd.DataFrame,
            column: str,
            mode: FrameFilteringMode,
            min_val: float | None = None,
            max_val: float | None = None,
            exact_val: float | str | None = None,
            func: Callable[[pd.Series], pd.Series] | None = None,
    ) -> pd.DataFrame:

        filter_types = [
            exact_val is not None,
            min_val is not None or max_val is not None,
            func is not None,
        ]

        if sum(filter_types) != 1:
            raise ValueError("Exactly one of exact_value, range, or func must be provided.")

        series = df[column]

        # ---------------- Callable filter (highest priority)
        if func is not None:
            """
            # EXAMPLES
            # Scalar/vectorised operations:
            func = lambda s: s == exact_val
            func = lambda s: s.between(min_val, max_val)
            func = lambda x: x.str.contains(".*FOO.*|.*DUMMY.*", na=False)  # no space between |
            #
            # Array/list operations:
            func = lambda s: s.apply(lambda arr: np.mean(arr) > 2)
            func = lambda s: s.apply(lambda arr: np.any(arr > 11) and np.all(arr > 11))  # non-empty only
            """
            row_mask = func(series)

            if not isinstance(row_mask, pd.Series):
                raise TypeError(f"'func' must return pandas Series, got {type(row_mask)}")

            if not row_mask.index.equals(series.index):
                raise ValueError("'func' must return Series aligned with input index")

            if row_mask.dtype != bool:
                raise TypeError("'func' must return boolean Series.")

        # ---------------- Fast vectorised scalar/string paths
        elif pd.api.types.is_string_dtype(series) and exact_val is not None:
            row_mask = series == exact_val
        elif pd.api.types.is_numeric_dtype(series) and exact_val is not None:
            row_mask = series == exact_val
        elif pd.api.types.is_numeric_dtype(series):  # numeric min/max
            low = -np.inf if min_val is None else min_val
            high = np.inf if max_val is None else max_val
            row_mask = series.between(low, high)

        # ---------------- Generic (arrays / objects)
        else:
            row_mask = series.apply(
                _cell_satisfies,
                min_val=min_val,
                max_val=max_val,
                exact_val=exact_val,
            )

        if row_mask.isna().any():
            raise ValueError(
                f"Non-boolean mask produced for column '{column}'. "
                f"Check NaNs or invalid cell values."
            )

        # ---------------- Frame-wise
        if mode == "frame-wise":
            return df[row_mask]

        # ---------------- Group-wise reduction
        if mode == "any":
            group_mask = row_mask.groupby(group_keys, observed=True).any()
        elif mode == "all":
            group_mask = row_mask.groupby(group_keys, observed=True).all()
        else:
            raise ValueError(f"Unknown mode '{mode}'")
        """
        if mode == "any":
            group_mask = row_mask.groupby(group_keys, observed=True).agg("any")
        elif mode == "all":
            group_mask = row_mask.groupby(group_keys, observed=True).agg(
                lambda x: x.any() and x.all()
            )
        else:
            raise ValueError(f"Unknown mode '{mode}'")
        """

        if group_mask.isna().any():
            raise ValueError(
                f"NaN encountered in group mask for column '{column}'."
            )

        keep_ids = group_mask[group_mask].index
        idx = pd.MultiIndex.from_frame(df[group_cols])

        return df[idx.isin(keep_ids)]

    # ------------------------------------------------------------------
    # Apply all filters sequentially
    # ------------------------------------------------------------------

    for key, spec in filtering_kwargs.items():

        # ---- Case 1: direct column
        if key in df:
            df = _apply_filter(
                df,
                column=key,
                min_val=spec.get("min_value"),
                max_val=spec.get("max_value"),
                exact_val=spec.get("exact_value"),
                mode=spec["mode"],
                func=spec.get("func"),
            )
            continue

        # ---- Case 2: structured
        part = key
        for param, p_spec in spec.items():
            col = _build_column_name(
                part=part,
                param=param,
                stats_key=p_spec.get("stats_key", "Ic"),
            )

            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame")

            df = _apply_filter(
                df,
                column=col,
                min_val=p_spec.get("min_value"),
                max_val=p_spec.get("max_value"),
                exact_val=p_spec.get("exact_value"),
                mode=p_spec["mode"],
                func=p_spec.get("func"),
            )

    return df
