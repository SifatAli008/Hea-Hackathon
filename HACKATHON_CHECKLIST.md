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

## 🔶 Add to Strengthen (Before Judging)

### 1. Fairness (judges check this)
- **Add:** Short "Fairness" note in README: e.g. "We did not use age/gender/ethnicity as model features; bias audit can be run when those variables are available in the dataset."
- **Optional:** If NLSY97 columns include demographics, add a notebook or app section that stratifies F2/errors by group and documents "no large disparity."

### 2. Feature / no-leakage audit (judges audit features)
- **Add:** In README, one sentence: "Features: baseline deviations, % change, z-scores, trend slopes, declining flags only; no medication or diagnosis-related variables."

### 3. Presentation (5 min)
- **Prepare:** Problem → Dataset (NLSY97/sample) → Approach (baseline + weak signals + no-leakage target) → Live demo (deploy link) → Impact (recall-focused, explainable, safe follow-up).

### 4. Optional: architecture diagram
- **Add:** Simple flow in README (e.g. Data → Baseline → Weak signals → Model → Score + Explain + Follow-up) so judges see the pipeline at a glance.

---

## One-line summary

You already meet the core rules and judging criteria. Adding a **fairness note**, **feature audit sentence**, and a **clear 5-min pitch** (plus optional diagram) will make the submission more complete and easier for judges to score.
