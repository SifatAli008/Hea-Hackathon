"""
Explainability: which features contributed, by how much; human-readable summary.
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict


def feature_importance_lr(model, feature_names: List[str]) -> pd.Series:
    """Logistic Regression: absolute coefficient as importance."""
    if not hasattr(model, "coef_"):
        return pd.Series(dtype=float)
    coef = np.abs(model.coef_.ravel())
    return pd.Series(coef, index=feature_names).sort_values(ascending=False)


def feature_importance_rf(model, feature_names: List[str]) -> pd.Series:
    """Random Forest: feature_importances_."""
    if not hasattr(model, "feature_importances_"):
        return pd.Series(dtype=float)
    return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)


def get_top_contributors(
    model,
    feature_names: List[str],
    model_type: str = "logistic",
    top_k: int = 5,
) -> pd.Series:
    """Top-k features that drove the model."""
    if model_type == "logistic":
        imp = feature_importance_lr(model, feature_names)
    else:
        imp = feature_importance_rf(model, feature_names)
    return imp.head(top_k)


def human_readable_changes(
    row: pd.Series,
    deviation_cols: Optional[List[str]] = None,
    pct_cols: Optional[List[str]] = None,
) -> List[str]:
    """
    Build list of sentences like "Sleep decreased 22%", "Mood score dropped 1.8 points".
    deviation_cols: columns ending with _deviation or _pct_change.
    """
    out = []
    if deviation_cols is None:
        deviation_cols = [c for c in row.index if "_pct_change" in c or "_deviation" in c]
    for c in deviation_cols:
        if c not in row.index or pd.isna(row[c]):
            continue
        val = row[c]
        name = c.replace("_pct_change", "").replace("_deviation", "").replace("_", " ").title()
        if "_pct_change" in c:
            if val > 0:
                out.append(f"{name} increased {val:.1f}%")
            else:
                out.append(f"{name} decreased {abs(val):.1f}%")
        else:
            if val > 0:
                out.append(f"{name} increased {val:.2f}")
            else:
                out.append(f"{name} decreased {abs(val):.2f}")
    return out[:5]  # top 5


def explanation_text(contributions: List[str], risk_score: float, category: str) -> str:
    """Single paragraph for user: why flagged + score + category."""
    lines = [
        f"Risk score: {risk_score:.0f}/100 ({category}).",
        "Main changes we observed: " + "; ".join(contributions) if contributions else "No strong deviations from your baseline.",
    ]
    return " ".join(lines)
