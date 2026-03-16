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
        slope_path: str,
        control_plots: bool = False
) -> pd.DataFrame:
    """
    Fit piecewise-linear models to total magnetic flux evolution
    for each spot and return segment-level statistics.
    """
    segments: list[dict] = []

    if is_empty(df):
        raise ValueError("No contour files at the input")

    if control_plots:
        from scr.config.paths import PATH_FIGURES
        from scr.statistics.segments.control_plots import plot_flux_fit_control

        fig_outdir = path.join(PATH_FIGURES, "flux_fit")
        check_dir(fig_outdir, is_file=False)

    for _, g in tqdm(df.groupby("spot_global_index", observed=True),
                     desc="Fitting piecewise linear slopes per spot",
                     unit="sunspot"
                     ):

        # ----------------------------------------------------------
        # Time axis + total flux
        # ----------------------------------------------------------

        t = g["frame"].to_numpy(dtype=float)
        total_flux = g["Br_sunspot_flux_corr_total"].to_numpy(dtype=float)

        # ----------------------------------------------------------
        # Finite / outlier handling
        # ----------------------------------------------------------

        total_flux = np.abs(total_flux)

        idx_finite = np.isfinite(total_flux)
        t, total_flux = t[idx_finite], total_flux[idx_finite]
        if np.sum(idx_finite) <= 1:
            continue

        total_flux[find_outliers1D(total_flux, t, max_iter=1)] = np.nan
        idx_finite = np.isfinite(total_flux)
        t, total_flux = t[idx_finite], total_flux[idx_finite]
        if np.sum(idx_finite) <= 1:
            continue

        # ----------------------------------------------------------
        # Normalisation + fitting
        # ----------------------------------------------------------

        flux_max = np.nanmax(total_flux)
        total_flux /= flux_max  # normalise
        model, results = fit_optimal_piecewise_linear_model(t, total_flux, verbose=False)

        if model is None:
            continue

        obs_id = g["observation_id"].iloc[0]
        sunspot_id = g["sunspot_id"].iloc[0]

        if control_plots:
            basename = path.basename(obs_id).replace(".npz", f"_{sunspot_id:04d}.jpg")
            plot_flux_fit_control(
                t=t,
                total_flux=total_flux,
                model=model,
                outfile=path.join(fig_outdir, basename),
                use_tex=False,
            )

        # ----------------------------------------------------------
        # Segment loop
        # ----------------------------------------------------------

        for i in range(len(model.fit_breaks) - 1):
            x0 = model.fit_breaks[i]
            x1 = model.fit_breaks[i + 1]
            slope = model.slopes[i]

            y0 = model.predict([x0])[0]
            y1 = model.predict([x1])[0]

            # total_flux = slope * t + intercept; t in [start; stop]
            intercept = y0 - slope * x0
            relative_slope = slope / y0 if y0 != 0. else np.nan

            segments.append({
                "observation_id": obs_id,
                "sunspot_id": sunspot_id,
                "segment_index": i,
                "start": x0,
                "stop": x1,
                "duration": x1 - x0,
                "slope": slope,
                "intercept": intercept,
                "flux_max": flux_max,
                "flux_start": y0,
                "flux_stop": y1,
                "mean_flux": 0.5 * (y0 + y1),
                "relative_slope": relative_slope
            })

    segments_df = pd.DataFrame(segments)

    # Optimise ID columns
    cols = ["observation_id", "sunspot_id", "segment_index"]
    segments_df[cols] = segments_df[cols].astype("category")

    cols = ["start", "stop", "duration", "slope", "intercept", "flux_max", "flux_start", "flux_stop", "mean_flux", "relative_slope"]
    segments_df[cols] = segments_df[cols].astype(WP)

    check_dir(slope_path, is_file=True)
    save_parquet(filename=slope_path, df=segments_df)

    return segments_df
