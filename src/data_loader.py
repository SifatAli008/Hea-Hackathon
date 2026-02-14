"""
Load and align longitudinal data. Handles missing values and noisy self-reports.
"""
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List

from .config import NLSY97_CSV, ID_COL, WAVE_COL, TIME_COL


def _get_header_first_columns(path: Path, max_cols: int = 50) -> List[str]:
    """Read first line of CSV to get column names; return first max_cols only (no pandas, minimal memory)."""
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        first_row = next(reader)
    return [c.strip().strip('"') for c in first_row[:max_cols]]


def _load_csv_line_by_line(path: Path, max_cols: int, n_rows: int) -> pd.DataFrame:
    """
    Read CSV line by line, keeping only first max_cols per row. No pandas parser — avoids OOM on huge wide files.
    """
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        header = [c.strip().strip('"') for c in next(reader)[:max_cols]]
        rows = []
        for i, row in enumerate(reader):
            if i >= n_rows:
                break
            rows.append(row[:max_cols])  # only first max_cols; pad if short
        # Pad short rows so all have len(header) columns
        n_cols = len(header)
        padded = [(r[:n_cols] + [""] * (n_cols - len(r)))[:n_cols] for r in rows]
    df = pd.DataFrame(padded, columns=header)
    # Coerce numeric where possible (first col often ID, rest may be numeric)
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_longitudinal(
    path: Optional[Path] = None,
    id_col: str = ID_COL,
    use_chunks: bool = False,
    chunk_rows: int = 100_000,
    sample_n: Optional[int] = None,
    max_cols: Optional[int] = 50,
) -> pd.DataFrame:
    """
    Load longitudinal CSV. Uses line-by-line read (no pandas parser) to avoid OOM on huge wide files (e.g. NLSY97).
    """
    path = path or NLSY97_CSV
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}. Set HEA_DATA_PATH or pass path.")

    n_to_read = min(sample_n or 1500, 2000)
    max_cols = max_cols or 50

    df = _load_csv_line_by_line(path, max_cols=max_cols, n_rows=n_to_read)

    if df is None or len(df) == 0:
        raise ValueError(f"No rows read from {path}")

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
