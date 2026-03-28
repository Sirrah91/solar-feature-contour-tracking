import numpy as np
import pandas as pd
from os import path
from tqdm import tqdm

from scr.config.numerics import WP

from scr.utils.filesystem import check_dir, is_empty
from scr.utils.numerics import find_outliers1D

from scr.io.parquet import save_parquet

from scr.statistics.segments.fitting import fit_optimal_piecewise_linear_model


def collect_slopes(
        df: pd.DataFrame,
        slope_path: str | None,
        control_plots: bool = False
) -> pd.DataFrame:
    """
    Memory-efficient slope collection without pandas groupby.
    """

    if is_empty(df):
        raise ValueError("No contour files at the input")

    # ----------------------------------------------------------
    # Optional plotting setup
    # ----------------------------------------------------------
    if control_plots:
        from scr.config.paths import PATH_FIGURES
        from scr.statistics.segments.control_plots import plot_flux_fit_control

        fig_outdir = path.join(PATH_FIGURES, "flux_fit")
        check_dir(fig_outdir, is_file=False)

    # ----------------------------------------------------------
    # Sort once → enables fast slicing instead of groupby
    # ----------------------------------------------------------
    df = df.sort_values("spot_global_index")

    spot_ids = df["spot_global_index"].to_numpy()
    frames = df["frame"].to_numpy(dtype=float)
    flux_all = df["Br_sunspot_flux_corr_total"].to_numpy(dtype=float)

    obs_ids = df["observation_id"].to_numpy()
    sunspot_ids = df["sunspot_id"].to_numpy()

    n = len(df)

    # ----------------------------------------------------------
    # Preallocate lists (faster than dict-per-row)
    # ----------------------------------------------------------
    out = {
        "observation_id": [],
        "sunspot_id": [],
        "segment_index": [],
        "start": [],
        "stop": [],
        "duration": [],
        "slope": [],
        "intercept": [],
        "flux_max": [],
        "flux_start": [],
        "flux_stop": [],
        "mean_flux": [],
        "relative_slope": [],
    }

    # ----------------------------------------------------------
    # Helper to process one spot slice
    # ----------------------------------------------------------
    def process_slice(i0: int, i1: int):
        t = frames[i0:i1]
        total_flux = flux_all[i0:i1]

        # --- preprocessing ---
        total_flux = np.abs(total_flux)

        mask = np.isfinite(total_flux)
        if mask.sum() <= 1:
            return

        t_ = t[mask]
        f_ = total_flux[mask]

        # remove outliers
        outliers = find_outliers1D(f_, t_, max_iter=1)
        if np.any(outliers):
            f_[outliers] = np.nan
            mask2 = np.isfinite(f_)
            if mask2.sum() <= 1:
                return
            t_ = t_[mask2]
            f_ = f_[mask2]

        # --- normalisation ---
        flux_max = np.nanmax(f_)
        if flux_max == 0 or not np.isfinite(flux_max):
            return

        f_ = f_ / flux_max

        # --- fit ---
        model, _ = fit_optimal_piecewise_linear_model(t_, f_, verbose=False)
        if model is None:
            return

        obs_id = obs_ids[i0]
        sunspot_id = sunspot_ids[i0]

        # optional plotting
        if control_plots:
            basename = path.basename(obs_id).replace(".npz", f"_{sunspot_id:04d}.jpg")
            plot_flux_fit_control(
                t=t_,
                total_flux=f_,
                model=model,
                outfile=path.join(fig_outdir, basename),
                use_tex=False,
            )

        # --- segments ---
        breaks = model.fit_breaks
        slopes = model.slopes

        for seg_idx in range(len(breaks) - 1):
            x0 = breaks[seg_idx]
            x1 = breaks[seg_idx + 1]
            slope = slopes[seg_idx]

            # direct computation instead of predict()
            # y = slope * t + intercept
            # compute intercept from first point
            y0 = model.predict([x0])[0]  # keep if model is complex
            intercept = y0 - slope * x0
            y1 = slope * x1 + intercept

            rel_slope = slope / y0 if y0 != 0.0 else np.nan

            out["observation_id"].append(obs_id)
            out["sunspot_id"].append(sunspot_id)
            out["segment_index"].append(seg_idx)
            out["start"].append(x0)
            out["stop"].append(x1)
            out["duration"].append(x1 - x0)
            out["slope"].append(slope)
            out["intercept"].append(intercept)
            out["flux_max"].append(flux_max)
            out["flux_start"].append(y0)
            out["flux_stop"].append(y1)
            out["mean_flux"].append(0.5 * (y0 + y1))
            out["relative_slope"].append(rel_slope)

    # ----------------------------------------------------------
    # Main loop (no groupby!)
    # ----------------------------------------------------------
    start = 0

    for i in tqdm(range(1, n), desc="Fitting slopes", unit="rows"):
        if spot_ids[i] != spot_ids[start]:
            process_slice(start, i)
            start = i

    # last slice
    process_slice(start, n)

    # ----------------------------------------------------------
    # Build DataFrame
    # ----------------------------------------------------------
    segments_df = pd.DataFrame(out)

    if segments_df.empty:
        return segments_df

    # --- dtype optimisation ---
    cat_cols = ["observation_id", "sunspot_id"]
    segments_df[cat_cols] = segments_df[cat_cols].astype("category")

    segments_df["segment_index"] = segments_df["segment_index"].astype(np.int16)

    float_cols = [
        "start", "stop", "duration", "slope", "intercept",
        "flux_max", "flux_start", "flux_stop",
        "mean_flux", "relative_slope"
    ]
    segments_df[float_cols] = segments_df[float_cols].astype(WP)

    # ----------------------------------------------------------
    # Save if requested
    # ----------------------------------------------------------
    if slope_path is not None:
        check_dir(slope_path, is_file=True)
        save_parquet(filename=slope_path, df=segments_df)

    return segments_df
