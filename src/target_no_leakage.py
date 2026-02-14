"""
No-leakage target and features: target from LAST wave only, features from PAST waves only.
Avoids circular "predict declining flags from declining flags".
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple

from .config import ID_COL, WAVE_COL, RANDOM_STATE


def build_no_leakage_training(
    df: pd.DataFrame,
    feature_cols: List[str],
    id_col: str = ID_COL,
    wave_col: str = WAVE_COL,
    target_threshold: float = 2.5,
    target_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    One row per person: features from waves 0..T-1 only, target from last wave T.
    Target = 1 if value at last wave < target_threshold (e.g. low health), else 0.
    No leakage: we never use last wave in features.
    """
    target_col = target_col or (feature_cols[0] if feature_cols else None)
    if not target_col or target_col not in df.columns:
        return pd.DataFrame(), pd.Series(dtype=int)

    rows = []
    y_list = []
    for pid, grp in df.groupby(id_col):
        grp = grp.sort_values(wave_col)
        waves = grp[wave_col].values
        if len(waves) < 3:
            continue
        last_wave = waves[-1]
        past = grp[grp[wave_col] < last_wave]
        if len(past) < 2:
            continue

        # Target from LAST wave only (future relative to features)
        last_row = grp[grp[wave_col] == last_wave].iloc[0]
        y_val = last_row[target_col]
        if pd.isna(y_val):
            continue
        y = 1 if float(y_val) < target_threshold else 0
        y_list.append(y)

        # Features from PAST waves only (no last wave)
        row = {id_col: pid}
        for f in feature_cols:
            if f not in past.columns:
                continue
            vals = past[f].values
            vals = np.nan_to_num(vals.astype(float), nan=np.nanmean(vals))
            baseline_mean = np.mean(vals)
            baseline_std = np.std(vals) or 1e-6
            current_val = vals[-1]
            deviation = current_val - baseline_mean
            pct_change = (deviation / (abs(baseline_mean) or 1e-6)) * 100
            z = (current_val - baseline_mean) / baseline_std
            x = np.arange(len(vals))
            slope = np.polyfit(x, vals, 1)[0]
            declining = 1 if slope < -0.05 else 0
            row[f"{f}_deviation"] = deviation
            row[f"{f}_pct_change"] = pct_change
            row[f"{f}_z"] = z
            row[f"{f}_slope"] = slope
            row[f"{f}_declining"] = declining
        rows.append(row)

    if not rows:
        return pd.DataFrame(), pd.Series(dtype=int)

    X_df = pd.DataFrame(rows)
    y_series = pd.Series(y_list, index=X_df.index)
    return X_df, y_series
