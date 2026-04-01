import pandas as pd

from scr.utils.types_alias import SunspotsPhasesByObservation

from scr.io.npz import load_npz
from scr.io.parquet import load_parquet
from scr.io.pickle import load_pickle


def load_sunspots_and_df_stat(
        nosuffix_filename: str
) -> tuple[SunspotsPhasesByObservation, pd.DataFrame]:
    """Load contours (.npz) and statistics (.parquet) sharing the same base name."""
    for suffix in [".npz", ".parquet"]:
        if nosuffix_filename.endswith(suffix):
            nosuffix_filename = nosuffix_filename[:-len(suffix)]

    contours = load_npz(f"{nosuffix_filename}.npz")["sunspots_phases"].item()
    df = load_parquet(f"{nosuffix_filename}.parquet", return_dataset=True)

    return contours, df


def load_sunspots_and_stat_pickle(
        filename: str,
) -> tuple[SunspotsPhasesByObservation, pd.DataFrame]:
    # sunspots_phases, combined_df
    return load_pickle(filename)
