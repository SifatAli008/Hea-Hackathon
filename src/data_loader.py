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
) -> pd.DataFrame:
    """
    Load longitudinal CSV. For very large files, use use_chunks=True or sample_n.
    """
    path = path or NLSY97_CSV
    if not Path(path).exists():
        raise FileNotFoundError(f"Data not found: {path}. Set HEA_DATA_PATH or pass path.")

    if use_chunks:
        chunks = []
        for chunk in pd.read_csv(path, chunksize=chunk_rows, low_memory=False):
            chunks.append(chunk)
            if sample_n and sum(len(c) for c in chunks) >= sample_n:
                break
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_csv(path, nrows=sample_n, low_memory=False)

    # Ensure we have a person id column (rename if codebook says so)
    if id_col not in df.columns and df.columns[0] is not None:
        # Common NLSY97: first column often person ID
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
