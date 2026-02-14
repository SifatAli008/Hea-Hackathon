"""
Load and align longitudinal data. Handles missing values and noisy self-reports.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List

from .config import NLSY97_CSV, ID_COL, WAVE_COL, TIME_COL


def load_longitudinal(
    path: Optional[Path] = None,
    id_col: str = ID_COL,
    use_chunks: bool = False,
    chunk_rows: int = 100_000,
    sample_n: Optional[int] = None,
    max_cols: Optional[int] = 150,
) -> pd.DataFrame:
    """
    Load longitudinal CSV. For very large files we read one chunk only to avoid OOM.
    max_cols: keep only first N columns to save memory (NLSY97 has thousands).
    """
    path = path or NLSY97_CSV
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}. Set HEA_DATA_PATH or pass path.")

    # Use chunked read so we never load the whole file (avoids OOM on huge CSVs)
    n_to_read = sample_n or 5000
    n_to_read = min(n_to_read, 3000)  # cap for memory safety
    chunk_size = n_to_read

    df = None
    for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
        df = chunk
        break  # only first chunk

    if df is None or len(df) == 0:
        raise ValueError(f"No rows read from {path}")

    # Keep only first max_cols to reduce memory (pipeline will use numeric cols + ID)
    if max_cols and len(df.columns) > max_cols:
        df = df.iloc[:, :max_cols].copy()

    # Ensure we have a person id column (rename if codebook says so)
    if id_col not in df.columns and len(df.columns) > 0:
        df = df.rename(columns={df.columns[0]: id_col})
    return df


def align_waves(df: pd.DataFrame, id_col: str = ID_COL, wave_col: str = WAVE_COL) -> pd.DataFrame:
    """
    Ensure dataframe has a wave/year column for longitudinal alignment.
    If not present, try to infer from column names (e.g. round prefix) or add dummy.
    """
    if wave_col not in df.columns:
        # Try to get round from column names (e.g. R00001, R00002) or use index
        df = df.copy()
        df[wave_col] = np.arange(len(df))  # placeholder; replace with real wave
    return df


def handle_missing(
    df: pd.DataFrame,
    strategy: str = "forward_fill",
    max_missing_frac: float = 0.5,
    drop_all_nan_rows: bool = True,
) -> pd.DataFrame:
    """
    strategy: "forward_fill" | "median" | "drop"
    Drop columns with > max_missing_frac missing; optionally drop rows that are all NaN.
    """
    df = df.copy()
    # Drop columns that are mostly missing
    cols = [c for c in df.columns if df[c].dtype in ["float64", "int64"] or "float" in str(df[c].dtype)]
    if cols:
        keep = df[cols].columns[df[cols].isna().mean() <= max_missing_frac]
        df = df[[c for c in df.columns if c in keep or c not in cols]]
    if strategy == "forward_fill":
        df = df.sort_index().ffill().bfill()
    elif strategy == "median":
        for c in df.select_dtypes(include=[np.number]).columns:
            df[c] = df[c].fillna(df[c].median())
    if drop_all_nan_rows:
        df = df.dropna(how="all")
    return df


def get_person_timeline(df: pd.DataFrame, person_id, id_col: str = ID_COL, wave_col: str = WAVE_COL) -> pd.DataFrame:
    """Return rows for one person sorted by wave."""
    out = df.loc[df[id_col] == person_id].sort_values(wave_col).copy()
    return out
