"""
Fairness: stratified metrics by demographic group. Demographics are never model inputs.
Surfaces possible bias (e.g. F2 gap across groups) when DEMOGRAPHIC_COLS are available.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from sklearn.metrics import fbeta_score, average_precision_score, roc_auc_score
import warnings


def stratified_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    groups: pd.Series,
    beta: float = 2.0,
) -> Dict[str, Any]:
    """
    Compute F2, PR-AUC, ROC-AUC overall and per group. Drop groups with <10 samples.
    Returns dict: overall, by_group, disparity (max - min F2 across groups).
    """
    out = {"overall": {}, "by_group": {}, "disparity": {}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out["overall"]["f2"] = float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0))
        out["overall"]["pr_auc"] = float(average_precision_score(y_true, y_proba)) if y_true.sum() > 0 else 0.0
        out["overall"]["roc_auc"] = float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.0

    groups = pd.Series(groups).dropna()
    if len(groups) != len(y_true):
        return out
    uniq = groups.unique()
    f2s = []
    for g in uniq:
        mask = groups == g
        if mask.sum() < 10:
            continue
        yt, yp, ypr = y_true[mask], y_pred[mask], y_proba[mask]
        f2 = fbeta_score(yt, yp, beta=beta, zero_division=0)
        pr_auc = average_precision_score(yt, ypr) if yt.sum() > 0 else 0.0
        roc_auc = roc_auc_score(yt, ypr) if len(np.unique(yt)) > 1 else 0.0
        out["by_group"][str(g)] = {"f2": float(f2), "pr_auc": float(pr_auc), "roc_auc": float(roc_auc), "n": int(mask.sum())}
        f2s.append(float(f2))
    if f2s:
        out["disparity"]["f2_max_min"] = float(max(f2s) - min(f2s))
    return out
