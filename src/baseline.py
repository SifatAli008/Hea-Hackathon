"""
Per-person baseline builder. Compare current values to own history, not population.
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict

from .config import ID_COL, WAVE_COL, HEALTH_LIFESTYLE_COLS


def build_baselines(
    df: pd.DataFrame,
    id_col: str = ID_COL,
    wave_col: str = WAVE_COL,
    feature_cols: Optional[List[str]] = None,
    baseline_waves: Optional[int] = None,
    min_waves: int = 2,
) -> pd.DataFrame:
    """
    For each person, compute baseline mean (and std) from early waves.
    baseline_waves: use first N waves for baseline; if None, use first half of each person's history.
    Returns dataframe with one row per person: id_col, and for each feature: {feat}_baseline_mean, {feat}_baseline_std.
    """
    feature_cols = feature_cols or [c for c in HEALTH_LIFESTYLE_COLS if c in df.columns]
    if not feature_cols:
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in [id_col, wave_col]][:10]

    out = []
    for pid, grp in df.groupby(id_col):
        grp = grp.sort_values(wave_col)
        n_waves = len(grp)
        if n_waves < min_waves:
            continue
        n_baseline = baseline_waves if baseline_waves else max(1, n_waves // 2)
        baseline_grp = grp.head(n_baseline)

        row = {id_col: pid}
        for f in feature_cols:
            if f not in baseline_grp.columns:
                continue
            vals = baseline_grp[f].dropna()
            if len(vals) == 0:
                continue
            row[f"{f}_baseline_mean"] = vals.mean()
            row[f"{f}_baseline_std"] = vals.std()
            if row[f"{f}_baseline_std"] == 0:
                row[f"{f}_baseline_std"] = 1e-6  # avoid div by zero
        out.append(row)

    return pd.DataFrame(out)


def current_vs_baseline(
    df: pd.DataFrame,
    baselines: pd.DataFrame,
    id_col: str = ID_COL,
    wave_col: str = WAVE_COL,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    For each row (person-wave), attach baseline and compute deviation.
    Adds columns: {feat}_current, {feat}_baseline_mean, {feat}_deviation, {feat}_pct_change, {feat}_z.
    """
    feature_cols = feature_cols or [c.replace("_baseline_mean", "") for c in baselines.columns if c.endswith("_baseline_mean")]
    df = df.merge(baselines, on=id_col, how="left")
    out = df.copy()

    for f in feature_cols:
        mean_col = f"{f}_baseline_mean"
        std_col = f"{f}_baseline_std"
        if f not in out.columns or mean_col not in out.columns:
            continue
        out[f"{f}_current"] = out[f]
        out[f"{f}_deviation"] = out[f] - out[mean_col]
        out[f"{f}_pct_change"] = np.where(out[mean_col] != 0, (out[f] - out[mean_col]) / out[mean_col].abs() * 100, 0)
        if std_col in out.columns:
            out[f"{f}_z"] = (out[f] - out[mean_col]) / out[std_col].replace(0, np.nan)
    return out
