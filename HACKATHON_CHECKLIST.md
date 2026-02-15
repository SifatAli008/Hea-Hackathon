# Hackathon Submission Checklist

## ✅ What You Have (Strong)

| Requirement | Status |
|-------------|--------|
| **Context-aware detection** | ✓ Personal baseline, weak signals (slope, deviation), trends |
| **Risk score 0–100** | ✓ In app and exports |
| **Risk category** | ✓ Cardiovascular / Metabolic / Psycho-emotional |
| **Empathetic follow-up** | ✓ Template-based, no diagnosis, supportive |
| **Data processing** | ✓ Missing (NLSY97 codes→NaN), noisy, longitudinal, per-person fill |
| **Model metrics** | ✓ F2, PR-AUC, ROC-AUC in app |
| **No data leakage** | ✓ Target from last wave only, features from past only; no disease-revealing vars |
| **Explainability** | ✓ "Why flagged" + which features changed |
| **Simple & open-source** | ✓ Logistic Regression, pandas/sklearn/streamlit, 80%+ public |
| **Working prototype** | ✓ Streamlit app + deploy link |
| **GitHub** | ✓ README, setup, sample data, NLSY97 support |

---

## ✅ Already added (strengthened)

- **Fairness:** README has Fairness section + "Could the model become biased?"; optional stratified metrics when `DEMOGRAPHIC_COLS` set.
- **Feature / no-leakage audit:** README states features (baseline deviations, % change, z-scores, trend slopes, declining flags only; no medication/diagnosis).
- **Architecture diagram:** README has "Architecture (high level)" flow.
- **One-line summary:** README opens with one-line value prop ("Detect health risk from your own baseline...").
- **Quick start:** README has "Quick start (30 seconds)" (try app, run locally, use your data).

## 🔶 Still to do (team action)

### Presentation (5 min)
- **Prepare:** Problem → Dataset (NLSY97/sample) → Approach (baseline + weak signals + no-leakage) → Live demo (deploy link) → Impact (recall-focused, explainable, safe follow-up).
- **Rehearse** so you stay within 5 minutes and the demo works.

### Optional
- **Notebook:** Add a short demo notebook in `notebooks/` (load sample, run pipeline, show metrics).
- **Tests:** Minimal `tests/test_pipeline.py` that runs `run_pipeline()` on synthetic data and checks result keys.

---

## One-line summary

You meet the core rules and judging criteria. **Fairness note**, **feature audit**, **one-liner**, and **quick start** are in README. Deliver a **clear 5-min pitch** and you're in strong shape.
