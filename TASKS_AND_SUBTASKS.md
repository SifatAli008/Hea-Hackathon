# Hea Hackathon — Tasks & Subtasks

Work breakdown for the longitudinal health risk detection prototype.  
**Setup:** Nebius compute (Jupyter at `http://89.169.121.184:8888`) + S3 bucket `hackathon-team-fabric3-6` under `/s3`. No local train/test.

---

## 1. Environment & Data Setup

| ID | Task | Subtasks |
|----|------|----------|
| **1.1** | **Nebius & S3 access** | • Confirm Jupyter login and `/s3` mount<br>• Configure AWS CLI for Nebius endpoint (`eu-north1`, endpoint_url)<br>• List bucket `hackathon-team-fabric3-6` and note data paths |
| **1.2** | **Repo & codebase** | • Clone/code in `https://github.com/SifatAli008/Hea-Hackathon`<br>• Add README with setup (Nebius, S3, Python deps)<br>• Decide folder structure (e.g. `data/`, `src/`, `notebooks/`, `models/`) |
| **1.3** | **Python environment** | • Create `requirements.txt` (pandas, numpy, scikit-learn, streamlit if needed)<br>• Install on Nebius instance; document in README |
| **1.4** | **Dataset** | • Confirm NLSY97 (or other) location (local copy vs `/s3`)<br>• Load sample (e.g. `nlsy97_all_1997-2019.csv`) and verify columns (PUBID, health, CV_* etc.) |

---

## 2. Data Processing & Exploration

| ID | Task | Subtasks |
|----|------|----------|
| **2.1** | **Load & align longitudinal data** | • Load full dataset (from `/s3` or local path)<br>• Map waves/years to a consistent time index<br>• Build per-person timelines (one row per person per wave, or long format) |
| **2.2** | **Handle missing & noisy data** | • Define strategy: drop vs impute (median/mode/forward-fill) per variable type<br>• Implement and document in code<br>• Sanity checks (no all-nan users, plausible value ranges) |
| **2.3** | **Exploratory analysis** | • Describe health-related variables (conditions, substance use, self-rated health if any)<br>• Plot distributions and missingness by wave<br>• Identify candidate target (e.g. incident poor mental health, chronic condition) **without leakage** |
| **2.4** | **Leakage audit** | • List every feature used in model<br>• Remove or exclude any that directly reveal outcome (e.g. medication for predicted disease)<br>• Document “no leakage” choices for judges |

---

## 3. Baseline & Weak-Signal Detection

| ID | Task | Subtasks |
|----|------|----------|
| **3.1** | **Baseline builder** | • For each user: define baseline window (e.g. first N waves or first half of history)<br>• Compute baseline stats: mean (and optionally std) for sleep, mood, activity, stress, BMI/health rating, etc.<br>• Store baseline per user (and per feature) for comparison |
| **3.2** | **Deviation features** | • Current value vs personal baseline: difference, % change, or z-score (current − baseline_mean) / baseline_std<br>• Implement for all chosen health/lifestyle variables |
| **3.3** | **Temporal trends** | • Moving averages over last K waves<br>• Slope (e.g. linear regression of value on time) over recent window<br>• Flag “declining” or “worsening” trends (e.g. mood down, activity down) |
| **3.4** | **Life-event / context (optional)** | • If data has job loss, retirement, marital status, stress events: create binary or categorical features<br>• Use as model inputs or for post-hoc explanation only (no leakage) |

---

## 4. Risk Scoring & Categorization

| ID | Task | Subtasks |
|----|------|----------|
| **4.1** | **Target definition** | • Define binary (or multi-class) target from data (e.g. incident condition by wave T+1)<br>• Ensure temporal split: only use past data to predict future<br>• Handle class imbalance (e.g. class_weight or resampling) |
| **4.2** | **Model training** | • Train lightweight model (e.g. Logistic Regression or Random Forest) on baseline + deviation + trend features<br>• Optimize for **F2-score** (recall > precision); tune threshold if needed<br>• Optional: small hyperparameter search (e.g. C, max_depth) |
| **4.3** | **Risk score 0–100** | • Map model output (e.g. probability or decision function) to 0–100 scale<br>• Define bands: e.g. 0–30 Low, 31–60 Moderate, 61–100 High<br>• Ensure score is consistent and documented |
| **4.4** | **Risk category** | • Assign category from signals: Cardiovascular / Metabolic / Psycho-emotional<br>• Rule-based: e.g. “if mood/stress dominant → Psycho-emotional; if activity/BMI dominant → Metabolic; if BP/heart dominant → Cardiovascular”<br>• Implement and test on sample outputs |

---

## 5. Explainability

| ID | Task | Subtasks |
|----|------|----------|
| **5.1** | **Feature importance** | • Use model’s built-in importance (e.g. coefficients for LR, feature_importances_ for RF)<br>• Rank and expose top-K features that drove the score |
| **5.2** | **Per-user explanation** | • For a flagged user: list which features contributed most (and direction: e.g. “mood dropped”, “sleep decreased”)<br>• Optional: simple SHAP or similar if time permits and stays lightweight |
| **5.3** | **Human-readable summary** | • e.g. “Sleep decreased 22%; mood score dropped 1.8 points; stress increased 15%”<br>• Integrate into prototype UI/output |

---

## 6. Empathetic Follow-Up Question

| ID | Task | Subtasks |
|----|------|----------|
| **6.1** | **Template design** | • 3–5 templates by risk category or by dominant signal (mood, sleep, activity, stress)<br>• No diagnosis, no treatment; only supportive, context-gathering questions |
| **6.2** | **Generator logic** | • Map top contributing features → choose template (e.g. mood decline → “We noticed changes in your mood recently…”)<br>• Fill placeholders if any; output single question string |
| **6.3** | **Validation** | • Review outputs for safety (no diagnosis/treatment); tone check |

---

## 7. Evaluation & Fairness

| ID | Task | Subtasks |
|----|------|----------|
| **7.1** | **Train/validation/test split** | • Time-based or person-based split (e.g. last wave = test) so no future leakage<br>• Run all training on Nebius; report metrics on held-out set |
| **7.2** | **Metrics** | • Compute **F2-score**, **PR-AUC**, **ROC-AUC** on test set<br>• Log in notebook or script; document in README |
| **7.3** | **Fairness checks** | • Stratify metrics (or error rates) by **age group**, **gender**, **ethnicity** (if available and allowed)<br>• Document findings; flag and fix large disparities if possible |

---

## 8. Prototype (Demo)

| ID | Task | Subtasks |
|----|------|----------|
| **8.1** | **Interface choice** | • Decide: Jupyter only **or** Streamlit/Flask/FastAPI<br>• Streamlit recommended for clear demo flow |
| **8.2** | **End-to-end flow** | • Input: upload dataset or select pre-loaded data<br>• Steps: process → build baselines → detect weak signals → score → category → explanation → follow-up question<br>• Output: risk score (0–100), category, explanation text, empathetic question |
| **8.3** | **Polish** | • Clear labels and short instructions<br>• Optional: 1–2 plots (e.g. trend over time, feature importance) |

---

## 9. Repository & Documentation

| ID | Task | Subtasks |
|----|------|----------|
| **9.1** | **README** | • Problem and goal<br>• Dataset used (e.g. NLSY97)<br>• Approach (baseline, weak signals, model, categories)<br>• How to run (Nebius, S3, `pip install -r requirements.txt`, run notebook or app)<br>• Where to find: data processing, model, explainability, follow-up generator |
| **9.2** | **Code quality** | • Clean, readable code; minimal dependencies; ≥80% open source<br>• No API keys or secrets in repo (use env or instructions only) |
| **9.3** | **Reproducibility** | • Fixed random seeds where applicable<br>• Steps to reproduce metrics from README |

---

## 10. Presentation (5 min)

| ID | Task | Subtasks |
|----|------|----------|
| **10.1** | **Slides / script** | • Problem (1 min)<br>• Dataset (e.g. NLSY97) (0.5 min)<br>• Approach (baseline + weak signals + model + explainability + follow-up) (1.5 min)<br>• Demo (1.5 min)<br>• Impact & fairness (0.5 min) |
| **10.2** | **Demo rehearsal** | • Run full flow on Nebius; stable connection and data path |

---

## Summary Checklist

- [ ] **1** Environment & data on Nebius + S3
- [ ] **2** Data processing & no leakage
- [ ] **3** Baseline + weak-signal features
- [ ] **4** Risk score 0–100 + category
- [ ] **5** Explainability
- [ ] **6** Empathetic follow-up question
- [ ] **7** F2, PR-AUC, ROC-AUC + fairness
- [ ] **8** Working prototype (Jupyter or Streamlit)
- [ ] **9** README + clean repo
- [ ] **10** 5-min presentation

---

*All training and testing on Nebius instance; data from S3 bucket. No local train/test.*
