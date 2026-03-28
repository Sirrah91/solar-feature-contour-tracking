import pandas as pd


def load_parquet(filename: str) -> pd.DataFrame:
    """Load a parquet file into a DataFrame."""
    return pd.read_parquet(filename)


def save_parquet(filename: str, df: pd.DataFrame) -> None:
    """Save a parquet file into a DataFrame."""
    df.to_parquet(filename, index=False)
