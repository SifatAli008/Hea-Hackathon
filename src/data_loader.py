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


def _load_nlsy97_wide_to_long(path: Path, n_rows: int, n_waves: int = 10, n_features: int = 5) -> pd.DataFrame:
    """
    NLSY97 is wide: one row per person, many columns. Reshape to long: cols 1..n_features = wave0, next n_features = wave1, ...
    Reads first 1 + n_waves*n_features columns. Col 0 = PUBID, then n_waves blocks of n_features.
    """
    n_cols = 1 + n_waves * n_features
    df = _load_csv_line_by_line(path, max_cols=n_cols, n_rows=n_rows)
    if df.empty or len(df.columns) < n_cols:
        return df
    df = df.rename(columns={df.columns[0]: ID_COL})
    # Feature names for pipeline (health_rating, stress_level, activity_level, etc.)
    feat_names = ["health_rating", "stress_level", "activity_level", "var_3", "var_4"][:n_features]
    if len(feat_names) < n_features:
        feat_names += [f"var_{i}" for i in range(len(feat_names), n_features)]
    long_rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        # Use row index as PUBID (one person per wide row); NLSY97 col 0 may be missing codes (-4,-5)
        pid = i
        for w in range(n_waves):
            start = 1 + w * n_features
            end = start + n_features
            if end > len(row):
                break
            r = {ID_COL: pid, WAVE_COL: w}
            for j, name in enumerate(feat_names):
                val = row.iloc[start + j]
                v = pd.to_numeric(val, errors="coerce")
                # NLSY97 missing/skip codes → NaN
                if pd.notna(v) and int(v) in (-5, -4, -3, -2, -1):
                    v = np.nan
                r[name] = v
            long_rows.append(r)
    out = pd.DataFrame(long_rows)
    # Replace any remaining NLSY97 missing codes in feature columns
    nlsy97_missing = [-5, -4, -3, -2, -1]
    for c in feat_names:
        if c in out.columns:
            out[c] = out[c].replace(nlsy97_missing, np.nan)
    return out


def load_longitudinal(
    path: Optional[Path] = None,
    id_col: str = ID_COL,
    use_chunks: bool = False,
    chunk_rows: int = 100_000,
    sample_n: Optional[int] = None,
    max_cols: Optional[int] = 50,
    use_nlsy97_format: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Load longitudinal CSV. Uses line-by-line read to avoid OOM on huge wide files (e.g. NLSY97).
    When path contains 'nlsy97' (or use_nlsy97_format=True): loads real NLSY97, reshapes wide→long.
    """
    path = path or NLSY97_CSV
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}. Set HEA_DATA_PATH or pass path.")

    is_nlsy97 = use_nlsy97_format if use_nlsy97_format is not None else ("nlsy97" in path.name.lower())
    if is_nlsy97:
        n_to_read = min(sample_n or 8984, 8984)  # all NLSY97 rows if not specified
        df = _load_nlsy97_wide_to_long(path, n_rows=n_to_read, n_waves=10, n_features=5)
    else:
        n_to_read = min(sample_n or 1500, 2000)
        max_cols = max_cols or 50
        df = _load_csv_line_by_line(path, max_cols=max_cols, n_rows=n_to_read)
        if id_col not in df.columns and len(df.columns) > 0:
            df = df.rename(columns={df.columns[0]: id_col})

    if df is None or len(df) == 0:
        raise ValueError(f"No rows read from {path}")
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
