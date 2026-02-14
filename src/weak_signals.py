"""
Weak-signal detection: moving average change, z-score vs personal history, trend slope.
"""
import pandas as pd
import numpy as np
from typing import List, Optional

from .config import ID_COL, WAVE_COL, HEALTH_LIFESTYLE_COLS


def moving_average_change(
    df: pd.DataFrame,
    id_col: str = ID_COL,
    wave_col: str = WAVE_COL,
    feature_cols: Optional[List[str]] = None,
    window: int = 3,
) -> pd.DataFrame:
    """
    For each person-wave, compare current value to moving average of previous `window` waves.
    Adds columns: {feat}_ma, {feat}_ma_change.
    """
    feature_cols = feature_cols or [c for c in HEALTH_LIFESTYLE_COLS if c in df.columns]
    if not feature_cols:
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in [id_col, wave_col]]

    out = df.copy()
    for f in feature_cols:
        if f not in out.columns:
            continue
        out[f"{f}_ma"] = out.groupby(id_col)[f].transform(lambda x: x.rolling(window, min_periods=1).mean())
        out[f"{f}_ma_change"] = out[f] - out[f"{f}_ma"]
    return out


def trend_slope(
    df: pd.DataFrame,
    id_col: str = ID_COL,
    wave_col: str = WAVE_COL,
    feature_cols: Optional[List[str]] = None,
    window: int = 6,
) -> pd.DataFrame:
    """
    Per person, fit linear slope over last `window` waves for each feature.
    Negative slope = declining (potential risk). Adds {feat}_slope.
    """
    feature_cols = feature_cols or [c for c in HEALTH_LIFESTYLE_COLS if c in df.columns]
    if not feature_cols:
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in [id_col, wave_col]]

    slopes = []
    for pid, grp in df.groupby(id_col):
        grp = grp.sort_values(wave_col).tail(window)
        if len(grp) < 2:
            continue
        x = np.arange(len(grp))
        row = {id_col: pid, wave_col: grp[wave_col].iloc[-1]}
        for f in feature_cols:
            if f not in grp.columns:
                continue
            y = grp[f].values
            if np.isnan(y).all():
                continue
            y = np.nan_to_num(y, nan=np.nanmean(y))
            slope = np.polyfit(x, y, 1)[0]
            row[f"{f}_slope"] = slope
        slopes.append(row)

    slope_df = pd.DataFrame(slopes)
    out = df.merge(slope_df, on=[id_col, wave_col], how="left")
    return out


def flag_declining(
    df: pd.DataFrame,
    feature_slope_cols: Optional[List[str]] = None,
    threshold: float = -0.1,
) -> pd.DataFrame:
    """
    Add binary flags for "declining" trend (slope < threshold).
    Adds {feat}_declining = 1 if slope < threshold else 0.
    """
    if feature_slope_cols is None:
        feature_slope_cols = [c for c in df.columns if c.endswith("_slope")]
    out = df.copy()
    for c in feature_slope_cols:
        if c not in out.columns:
            continue
        out[c.replace("_slope", "_declining")] = (out[c] < threshold).astype(int)
    return out
