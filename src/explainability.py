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


def _base_name(col: str) -> str:
    """Base variable name without _deviation / _pct_change."""
    return col.replace("_pct_change", "").replace("_deviation", "").replace("_", " ").strip().title()


def human_readable_changes(
    row: pd.Series,
    deviation_cols: Optional[List[str]] = None,
    pct_cols: Optional[List[str]] = None,
) -> List[str]:
    """
    Build list of sentences like "Sleep decreased 22%", "Health Rating: −0.5 (−2.1%)".
    Deduplicates by variable: one line per variable, combining absolute and % when both exist.
    """
    if deviation_cols is None:
        deviation_cols = [c for c in row.index if "_pct_change" in c or "_deviation" in c]
    # Group by base name: collect (deviation_val, pct_val) per variable
    by_name: Dict[str, tuple] = {}
    for c in deviation_cols:
        if c not in row.index or pd.isna(row[c]):
            continue
        val = row[c]
        name = _base_name(c)
        if name not in by_name:
            by_name[name] = (None, None)
        dev, pct = by_name[name]
        if "_pct_change" in c:
            by_name[name] = (dev, val)
        else:
            by_name[name] = (val, pct)
    out = []
    for name, (dev, pct) in by_name.items():
        if dev is not None and pct is not None:
            dev_str = f"{dev:+.2f}" if dev != 0 else "0.00"
            pct_str = f"{pct:+.1f}%" if pct != 0 else "0.0%"
            out.append(f"{name}: {dev_str} ({pct_str})")
        elif pct is not None:
            if pct > 0:
                out.append(f"{name} increased {pct:.1f}%")
            else:
                out.append(f"{name} decreased {abs(pct):.1f}%")
        elif dev is not None:
            if dev > 0:
                out.append(f"{name} increased {dev:.2f}")
            else:
                out.append(f"{name} decreased {abs(dev):.2f}")
    return out[:5]


def main_change_names(
    row: pd.Series,
    deviation_cols: Optional[List[str]] = None,
) -> List[str]:
    """
    Return base variable names ordered by magnitude of change (biggest first).
    Used to align follow-up question with the main change in the explanation.
    """
    if deviation_cols is None:
        deviation_cols = [c for c in row.index if "_pct_change" in c or "_deviation" in c]
    by_name: Dict[str, float] = {}
    for c in deviation_cols:
        if c not in row.index or pd.isna(row[c]):
            continue
        val = row[c]
        name = _base_name(c)
        mag = abs(val)
        by_name[name] = max(by_name.get(name, 0), mag)
    return [n for n, _ in sorted(by_name.items(), key=lambda x: -x[1])][:5]


def explanation_text(contributions: List[str], risk_score: float, category: str, include_score: bool = False) -> str:
    """Single paragraph for user. If include_score=False (default), only main changes (score shown separately in UI)."""
    main = "; ".join(contributions) if contributions else "No strong deviations from your baseline."
    text = "Main changes we observed: " + main
    if include_score:
        text = f"Risk score: {risk_score:.0f}/100 ({category}). " + text
    return text
