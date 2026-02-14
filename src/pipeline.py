"""
End-to-end pipeline: load → baseline → weak signals → risk model → explain → follow-up.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from .data_loader import load_longitudinal, align_waves, handle_missing
from .baseline import build_baselines, current_vs_baseline
from .weak_signals import moving_average_change, trend_slope, flag_declining
from .risk_model import train_risk_model, score_0_100, risk_band, risk_category_from_signals
from .explainability import get_top_contributors, human_readable_changes, explanation_text, main_change_names
from .follow_up import pick_follow_up
from .target_no_leakage import build_no_leakage_training
from .config import ID_COL, WAVE_COL, HEALTH_LIFESTYLE_COLS, DEMOGRAPHIC_COLS, RANDOM_STATE
from . import fairness as fairness_mod
from sklearn.model_selection import train_test_split


def _synthetic_longitudinal(n_persons: int = 80, waves: int = 6) -> pd.DataFrame:
    """Small synthetic longitudinal data for demo when no CSV available."""
    np.random.seed(RANDOM_STATE)
    n = n_persons * waves
    df = pd.DataFrame({
        ID_COL: np.repeat(np.arange(n_persons), waves),
        WAVE_COL: np.tile(np.arange(waves), n_persons),
        "health_rating": np.clip(3 + np.cumsum(np.random.randn(n).reshape(n_persons, waves), axis=1).ravel() + np.repeat(np.random.randn(n_persons) * 0.3, waves), 1, 5),
        "stress_level": np.clip(2 + np.random.randn(n) * 0.4, 0, 5),
        "activity_level": np.clip(3 - np.cumsum(np.random.randn(n).reshape(n_persons, waves) * 0.08, axis=1).ravel(), 0, 5),
    })
    return df


def run_pipeline(
    data_path: Optional[Path] = None,
    sample_n: Optional[int] = 5000,
    target_col: Optional[str] = None,
    feature_cols: Optional[list] = None,
    df_preloaded: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Run full pipeline on data. If target_col is None, a synthetic target is used for demo.
    If data_path is None and df_preloaded is None, uses tiny synthetic data so demo runs without CSV.
    Returns dict with: model, baselines, metrics, and functions to score new rows.
    """
    # 1) Load
    if df_preloaded is not None:
        df = df_preloaded.copy()
    elif data_path and Path(data_path).exists():
        df = load_longitudinal(path=data_path, sample_n=sample_n)
    else:
        df = _synthetic_longitudinal()
    df = align_waves(df)
    # NLSY97: many -4/-5 → NaN, so use high max_missing_frac to keep columns
    is_nlsy97 = data_path and Path(data_path).exists() and "nlsy97" in str(data_path).lower()
    max_missing = 0.95 if is_nlsy97 else 0.5
    df = handle_missing(df, max_missing_frac=max_missing)
    # Per-person fill for longitudinal (NLSY97): fill NaN within each person only
    if ID_COL in df.columns and WAVE_COL in df.columns:
        df = df.sort_values([ID_COL, WAVE_COL])
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            df[num_cols] = df.groupby(ID_COL)[num_cols].ffill().bfill()
        df = df.dropna(how="all")

    feature_cols = feature_cols or [c for c in HEALTH_LIFESTYLE_COLS if c in df.columns]
    if not feature_cols:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric if c not in [ID_COL, WAVE_COL]][:8]
    if not feature_cols:
        raise ValueError("No feature columns left after loading; check data and handle_missing.")
    fairness_result = None

    # 2) Baselines
    baselines = build_baselines(df, feature_cols=feature_cols)
    df = current_vs_baseline(df, baselines, feature_cols=feature_cols)

    # 3) Weak signals
    df = moving_average_change(df, feature_cols=feature_cols)
    df = trend_slope(df, feature_cols=feature_cols, window=4)
    df = flag_declining(df)

    # 4) No-leakage training: target from LAST wave only, features from PAST waves only
    target_col_use = target_col or (feature_cols[0] if feature_cols else None)
    X_noleak, y_noleak = build_no_leakage_training(
        df, feature_cols=feature_cols,
        target_threshold=2.5,
        target_col=target_col_use,
    )
    if X_noleak.empty or len(y_noleak) < 10:
        # Fallback if too few persons: use old target (less strict)
        target_col_use = target_col or (feature_cols[0] if feature_cols else None)
        if target_col_use and target_col_use in df.columns:
            y = (df.groupby(ID_COL)[target_col_use].transform("last") < 2.5).astype(int)
        else:
            decline_cols = [c for c in df.columns if c.endswith("_declining")]
            y = df[decline_cols].sum(axis=1) > 0 if decline_cols else pd.Series(0, index=df.index)
            y = y.astype(int)
        model_feat = [c for c in df.columns if any(x in c for x in ["_z", "_deviation", "_pct_change", "_slope", "_declining"]) and c != ID_COL]
        model_feat = [c for c in model_feat if c in df.columns][:20]
        X = df[model_feat].fillna(0)
        model, threshold, f2, pr_auc, roc_auc, scaler = train_risk_model(X, y, model_type="logistic", test_size=0.2)
    else:
        model_feat = [c for c in X_noleak.columns if c != ID_COL and any(x in c for x in ["_deviation", "_pct_change", "_z", "_slope", "_declining"])]
        model_feat = [c for c in model_feat if c in X_noleak.columns][:20]
        X_train = X_noleak[model_feat].fillna(0)
        model, threshold, f2, pr_auc, roc_auc, scaler = train_risk_model(X_train, y_noleak, model_type="logistic", test_size=0.2)
        # For scoring full df we need same feature names; full df has them from step 3
        X = df[[c for c in model_feat if c in df.columns]].reindex(columns=model_feat).fillna(0)

    # 5b) Optional fairness: stratified metrics by demographic group (never used as features)
    fairness_result = None
    if DEMOGRAPHIC_COLS and all(c in df.columns for c in DEMOGRAPHIC_COLS) and not X_noleak.empty:
        demo_per_person = df.groupby(ID_COL)[DEMOGRAPHIC_COLS].first()
        group_col = DEMOGRAPHIC_COLS[0]
        groups_noleak = X_noleak[ID_COL].map(demo_per_person[group_col])
        X_noleak_feat = X_noleak[model_feat].fillna(0)
        _, X_te, _, y_te = train_test_split(
            X_noleak_feat, y_noleak, test_size=0.2, random_state=RANDOM_STATE,
            stratify=y_noleak if y_noleak.nunique() > 1 else None,
        )
        if scaler is not None:
            X_te_scaled = pd.DataFrame(scaler.transform(X_te), index=X_te.index, columns=X_te.columns)
        else:
            X_te_scaled = X_te
        probs_te = model.predict_proba(X_te_scaled)[:, 1]
        preds_te = (probs_te >= threshold).astype(int)
        groups_te = groups_noleak.loc[y_te.index]
        fairness_result = fairness_mod.stratified_metrics(
            y_te.values, preds_te, probs_te, groups_te, beta=2.0,
        )

    # 6) Score 0-100 and category for full df (for display); model was trained on no-leakage set
    X_score = df.reindex(columns=model_feat).fillna(0)
    if scaler is not None:
        X_score = pd.DataFrame(scaler.transform(X_score), index=X_score.index, columns=X_score.columns)
    probs = model.predict_proba(X_score)[:, 1]
    df["_risk_prob"] = probs
    df["risk_score"] = df["_risk_prob"].apply(score_0_100)
    df["risk_band"] = df["risk_score"].apply(risk_band)
    df["risk_category"] = df.apply(lambda r: risk_category_from_signals(r, psycho_cols=feature_cols), axis=1)

    top_contrib = get_top_contributors(model, model_feat, model_type="logistic", top_k=5)

    def score_one(row: pd.Series, _scaler=scaler) -> Tuple[float, str, str, str, str]:
        X_row = row.reindex(model_feat).fillna(0).values.reshape(1, -1)
        if _scaler is not None:
            X_row = _scaler.transform(X_row)
        prob = model.predict_proba(X_row)[0, 1]
        score = score_0_100(prob)
        band = risk_band(score)
        cat = risk_category_from_signals(row, psycho_cols=feature_cols)
        contrib_names = top_contrib.index.tolist()
        dev_cols = [c for c in row.index if "_pct_change" in c or "_deviation" in c]
        changes = human_readable_changes(row, deviation_cols=dev_cols)
        expl = explanation_text(changes, score, cat)
        main_names = main_change_names(row, deviation_cols=dev_cols)
        follow_up = pick_follow_up(contrib_names, cat, score, main_change_names=main_names)
        return score, band, cat, expl, follow_up

    out = {
        "model": model,
        "scaler": scaler,
        "baselines": baselines,
        "model_feat": model_feat,
        "threshold": threshold,
        "metrics": {"f2": f2, "pr_auc": pr_auc, "roc_auc": roc_auc},
        "score_one": score_one,
        "df": df,
    }
    if fairness_result is not None:
        out["fairness"] = fairness_result
    return out


def demo_with_synthetic():
    """Run pipeline on tiny synthetic data when no CSV is available."""
    np.random.seed(RANDOM_STATE)
    n = 200
    df = pd.DataFrame({
        ID_COL: np.repeat(np.arange(40), 5),
        WAVE_COL: np.tile(np.arange(5), 40),
        "health_rating": np.clip(3 + np.cumsum(np.random.randn(n) * 0.1).reshape(40, 5).ravel() + np.repeat(np.random.randn(40) * 0.5, 5), 1, 5),
        "stress_level": np.clip(2 + np.random.randn(n) * 0.5, 0, 5),
        "activity_level": np.clip(3 - np.cumsum(np.random.randn(n) * 0.05).reshape(40, 5).ravel(), 0, 5),
    })
    return run_pipeline(data_path=None, sample_n=None, feature_cols=["health_rating", "stress_level", "activity_level"])
