import pyarrow.dataset as ds
import pandas as pd


def load_parquet(
        filename: str,
        *,
        return_dataset: bool = False
) -> pd.DataFrame | ds.FileSystemDataset:
    """Load a parquet file into a DataFrame or ds.dataset."""
    if return_dataset:
        return ds.dataset(filename, format="parquet")
    else:
        return pd.read_parquet(filename)


def save_parquet(filename: str, df: pd.DataFrame) -> None:
    """Save a parquet file into a DataFrame."""
    df.to_parquet(filename, index=False)
