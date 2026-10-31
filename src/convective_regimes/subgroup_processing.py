from os import path
import pandas as pd
import numpy as np

from src.config.filtering import gimme_filtering_kwargs
from src.geometry.solar.units import pixelarea_to_Mm2
from src.io.parquet import load_parquet
from src.utils.filesystem import is_empty
from src.statistics.dataframe.filtering import filter_dataset_by_spots
from src.penumbra_evolution.dataframe import extract_completeness_only

from src.utils.logging import get_logger

logger = get_logger("convective_regimes.subgroup_processing")


PATH_SUNSPOTS_PHASES = "/nfsscratch/david/Contours/sunspots_removed_phases"

df = load_parquet(
    path.join(PATH_SUNSPOTS_PHASES, f"sunspots_phases.parquet"),
    return_dataset=True
)
df = filter_dataset_by_spots(df, gimme_filtering_kwargs("sunspots"))
df = df[["observation_id", "sunspot_id", "frame", "sunspot_area_corr", "phase", "spot_global_index", "Ic<0.5_topology-components"]]
df["area"] = pixelarea_to_Mm2(df["sunspot_area_corr"])


def per_sunspot_area(df: pd.DataFrame) -> pd.Series:
    """
    One area value per unique sunspot: max area during the stable phase
    if the sunspot reaches one, otherwise max area over its whole
    lifetime (forming/decaying only).
    """
    stable_max = (
        df[df["phase"] == "stable"]
        .groupby("spot_global_index")["area"]
        .max()
    )
    overall_max = df.groupby("spot_global_index")["area"].max()

    area = overall_max.copy()
    area.loc[stable_max.index] = stable_max
    return area.rename("area_characteristic")


def find_light_bridge_keys(
        data_all: list,
        *,
        boundary: str = "b605.0",
        gamma_max: float = 40.0,
        Bver_range: tuple[float, float] = (800.0, 1200.0),
        Bhor_max: float = 900.0,
        ic_range: tuple[float, float] = (0.5, 0.9),
        frac_thresh: float = 0.15,
        min_pixels: int = 5,
) -> set[tuple[int, int]]:
    """
    Identify (spot_global_index, frame) pairs whose intermediate-intensity
    population contains a light-bridge-like signature, using the same
    pixel-level criterion as the yellow contours in Fig. transition_example:
    gamma < gamma_max, Bver in Bver_range, Bhor < Bhor_max, restricted to
    intermediate-intensity pixels (ic_range).

    A frame is flagged if the fraction of its intermediate-intensity pixels
    satisfying this criterion exceeds frac_thresh, AND the absolute number
    of matching pixels is at least min_pixels (avoids flagging frames based
    on a couple of noisy pixels).
    """
    keys = set()

    for data in data_all:
        sgi_field = data[f"spot_global_index_{boundary}"]
        frame_field = data[f"frame_{boundary}"]
        Ic_field = data[f"Ic_{boundary}"]
        Br_field = data[f"Br_{boundary}"]       # Bver, in local-frame convention used elsewhere
        Bhor_field = data[f"Bhor_{boundary}"]
        Binc_field = data[f"Binc_{boundary}"]

        for sgi_arr, frame_arr, ic_arr, br_arr, bhor_arr, binc_arr in zip(
                sgi_field, frame_field, Ic_field, Br_field, Bhor_field, Binc_field
        ):
            if is_empty(sgi_arr):
                continue

            ic = np.abs(ic_arr)
            bver = np.abs(br_arr)
            bhor = np.abs(bhor_arr)
            gamma = np.rad2deg(binc_arr)

            intermediate = (ic > ic_range[0]) & (ic <= ic_range[1])
            n_intermediate = int(intermediate.sum())
            if n_intermediate == 0:
                continue

            lb_like = (
                intermediate
                & (gamma < gamma_max)
                & (bver > Bver_range[0]) & (bver < Bver_range[1])
                & (bhor < Bhor_max)
            )
            n_lb = int(lb_like.sum())

            if n_lb >= min_pixels and (n_lb / n_intermediate) >= frac_thresh:
                keys.add((int(sgi_arr[0]), int(frame_arr[0])))

    return keys


area_per_spot = per_sunspot_area(df)
large_thresh = area_per_spot.quantile(0.75)   # 842.7 Mm^2
small_thresh = area_per_spot.quantile(0.25)   # 387.2 Mm^2

large_spots = area_per_spot[area_per_spot > large_thresh].index
small_spots = area_per_spot[area_per_spot < small_thresh].index

# small: all frames are already small by construction
df_small = df[df["spot_global_index"].isin(small_spots)]

# large: additionally require the frame itself to be large
# (avoids including formation/decay frames that were still small)
df_large = df[
    df["spot_global_index"].isin(large_spots)
    & (df["area"] > large_thresh)
]

df_comp = extract_completeness_only(
    "/nfsscratch/david/Contours/sunspots_removed_phases/completeness_results_Ic<0.9_Ic<0.5.npz",
    ramp_low=1.0, ramp_high=1.5,
)

df_large_complete = df_large.merge(
    df_comp[["observation_id", "sunspot_id", "frame", "object_completeness"]],
    on=["observation_id", "sunspot_id", "frame"],
    how="inner",
)
df_large_complete = df_large_complete[df_large_complete["object_completeness"] > 1./2.]

df_small_complete = df_small.merge(
    df_comp[["observation_id", "sunspot_id", "frame", "object_completeness"]],
    on=["observation_id", "sunspot_id", "frame"],
    how="inner",
)
df_small_complete = df_small_complete[df_small_complete["object_completeness"] > 1./2.]

from src.convective_regimes.io.loaders import load_all, extract_q_filtered, build_survivor_keys
from src.convective_regimes.analysis.probability import analyse_penumbra_probability
from src.convective_regimes.settings import DATA_DIR
from src.io.npz import save_npz
from src.io.parquet import save_parquet
from src.convective_regimes.io.filenames import regression_filename, probability_filename

BOUNDARY = "b605.0"
data_all = load_all(["sunspots"], ["all"])

df_split_umbra = df[df["Ic<0.5_topology-components"] >= 2]
split_umbra_keys = build_survivor_keys(df_split_umbra)

# magnetic signal: frames with the light-bridge-like pixel signature
magnetic_lb_keys = find_light_bridge_keys(data_all, boundary=BOUNDARY, frac_thresh=0.1)

# combined: both signals agree
light_bridge_keys = split_umbra_keys & magnetic_lb_keys

groups = {
    "small": build_survivor_keys(df_small),
    "small_complete": build_survivor_keys(df_small_complete),
    "large": build_survivor_keys(df_large),
    "large_complete": build_survivor_keys(df_large_complete),
    "light_bridge": light_bridge_keys,
}

from src.convective_regimes.plotting.probability import plot_probability_counts
from src.convective_regimes.plotting.regression import plot_regression
from src.convective_regimes.analysis.logistic import sliding_gamma_analysis

window_half_width = 5.0
min_samples = 1000
gamma_min = 5.0
gamma_max = 86.0
gamma_step = 2.0
gamma_centers = np.arange(gamma_min, gamma_max, gamma_step)

for name, keys in groups.items():
    logger.info(f"{name}: {len(keys)} (spot, frame) keys")

    gamma = extract_q_filtered(data_all, q="Binc", keys=keys, boundary=BOUNDARY)
    Ic = extract_q_filtered(data_all, q="Ic", keys=keys, boundary=BOUNDARY)
    Bhor = extract_q_filtered(data_all, q="Bhor", keys=keys, boundary=BOUNDARY)
    Bver = extract_q_filtered(data_all, q="Br", keys=keys, boundary=BOUNDARY)

    pmap = analyse_penumbra_probability(
        B=Bhor, gamma=gamma, Ic=Ic,
        ic_penumbra=(0.5, 0.9),
        min_count=50,
        n_B_bins=150, n_g_bins=45,
    )
    filename = probability_filename(
        data_dir=DATA_DIR, object_type="sunspots",
        phase="all", region="penumbra", quantity="Bhor",
        suf=f"_{name}",
    )

    save_npz(
        filename,
        probability=pmap.probability, counts=pmap.counts,
        x_bins=pmap.x_bins, y_bins=pmap.y_bins,
    )
    try:
        plot_probability_counts(suf=f"_{name}")
    except Exception as e:
        logger.error(e)

    for ic_boundary, region in ((0.5, "UP"), (0.9, "PQ")):
        results = sliding_gamma_analysis(
            gamma=gamma,
            bver=Bver,
            bhor=Bhor,
            target=(Ic > ic_boundary).astype(int),
            gamma_centers=gamma_centers,
            window_half_width=window_half_width,
            min_samples=min_samples,
        )

        filename = regression_filename(
            data_dir=DATA_DIR,
            object_type="sunspots",
            phase="all",
            region=region,
            suf=f"_{name}"
        )

        save_parquet(
            filename,
            results
        )

    try:
        plot_regression(suf=f"_{name}")
    except Exception as e:
        logger.error(e)
