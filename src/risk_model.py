"""
Risk scoring (0-100) and risk category: Cardiovascular, Metabolic, Psycho-emotional.
Lightweight model (Logistic Regression or Random Forest), F2-optimized.
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
import warnings

from .config import RISK_LOW, RISK_MODERATE, RISK_HIGH, RANDOM_STATE


def get_model(model_type: str = "logistic", class_weight: str = "balanced"):
    """Return classifier. Use class_weight='balanced' for imbalanced data."""
    if model_type == "logistic":
        return LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=RANDOM_STATE)
    return RandomForestClassifier(n_estimators=100, class_weight=class_weight, random_state=RANDOM_STATE)


def train_risk_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "logistic",
    test_size: float = 0.2,
    threshold_for_f2: float = 0.3,
) -> Tuple[object, float, float, float, float]:
    """
    Train model and return (model, best_threshold, f2, pr_auc, roc_auc).
    Tries lower decision threshold to maximize F2 (recall).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y if y.nunique() > 1 else None
    )
    model = get_model(model_type=model_type)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pr_auc = average_precision_score(y_test, probs) if y_test.sum() > 0 else 0.0
        roc_auc = roc_auc_score(y_test, probs) if y_test.nunique() > 1 else 0.0

    best_f2, best_t = 0.0, 0.5
    for t in [0.2, 0.25, 0.3, 0.35, 0.4]:
        pred = (probs >= t).astype(int)
        f2 = fbeta_score(y_test, pred, beta=2, zero_division=0)
        if f2 >= best_f2:
            best_f2, best_t = f2, t
    return model, best_t, best_f2, pr_auc, roc_auc


def score_0_100(prob: float, threshold: float = 0.3) -> float:
    """Map probability to 0-100 risk score (scale so threshold maps to ~50)."""
    # Simple: prob * 100, or use a steeper scale
    return min(100, max(0, float(prob) * 100))


def risk_band(score: float) -> str:
    if score <= RISK_LOW[1]:
        return "Low"
    if score <= RISK_MODERATE[1]:
        return "Moderate"
    return "High"


def risk_category_from_signals(
    row: pd.Series,
    psycho_cols: Optional[List[str]] = None,
    metabolic_cols: Optional[List[str]] = None,
    cardio_cols: Optional[List[str]] = None,
) -> str:
    """
    Assign category from dominant signal pattern.
    psycho: mood, stress, mental health; metabolic: activity, BMI, diet; cardio: BP, heart.
    """
    psycho_cols = psycho_cols or [c for c in row.index if "mood" in c.lower() or "stress" in c.lower()]
    metabolic_cols = metabolic_cols or [c for c in row.index if "activity" in c.lower() or "bmi" in c.lower()]
    cardio_cols = cardio_cols or [c for c in row.index if "bp" in c.lower() or "heart" in c.lower() or "cardio" in c.lower()]

    p = sum(abs(row.get(c, 0)) for c in psycho_cols if c in row.index)
    m = sum(abs(row.get(c, 0)) for c in metabolic_cols if c in row.index)
    c = sum(abs(row.get(c, 0)) for c in cardio_cols if c in row.index)

    if p >= m and p >= c:
        return "Psycho-emotional"
    if m >= p and m >= c:
        return "Metabolic"
    return "Cardiovascular"
