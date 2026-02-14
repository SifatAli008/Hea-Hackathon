# Hea — Personal Health Drift Detector (PHDD)

Longitudinal health risk detection from self-reported data. Detects **personal baseline drift** and **weak signals**, outputs a **risk score (0–100)**, **risk category**, **explanation**, and an **empathetic follow-up question**. No diagnosis, no treatment advice.

**Live app:** [https://hea-hackathon.streamlit.app/](https://hea-hackathon.streamlit.app/)

## Requirements

- Python 3.9+
- Run on **Nebius** instance (training/testing not on local machine). Data from S3 bucket or local path.

## Setup

```bash
pip install -r requirements.txt
```

On Nebius, data may be under `/s3` (S3 bucket `hackathon-team-fabric3-6`). Set path in app or:

```bash
export HEA_DATA_PATH=/s3/hackathon-team-fabric3-6
```

## Repo structure

```
Hea/
├── app.py                 # Streamlit demo
├── requirements.txt
├── README.md
├── TASKS_AND_SUBTASKS.md
├── src/
│   ├── config.py          # Paths, column names, constants
│   ├── data_loader.py     # Load longitudinal CSV, handle missing
│   ├── baseline.py        # Per-person baseline builder
│   ├── weak_signals.py     # Moving average, trend slope, flags
│   ├── risk_model.py      # Score 0–100, risk category, F2-optimized
│   ├── explainability.py  # Feature importance, human-readable summary
│   ├── follow_up.py       # Empathetic question templates
│   └── pipeline.py        # End-to-end pipeline
├── data/
│   └── sample_longitudinal.csv   # Small sample (in repo) — demo without NLSY97
├── HACKATHON_CHECKLIST.md  # Submission checklist vs judging criteria
├── RULES_GAP.md           # Rules.pdf — what's missing / done
└── notebooks/             # Optional exploration
```

## Run demo

**Deployed app:** [https://hea-hackathon.streamlit.app/](https://hea-hackathon.streamlit.app/)

**Local (Streamlit):**

```bash
streamlit run app.py
```

**Synthetic demo (no data file):**  
Use “Synthetic demo (no file)” in the sidebar — runs on small generated data.

**With data:**  
- Upload a CSV in the app, or  
- Set “Use path” to your NLSY97 path (e.g. `/s3/.../nlsy97_all_1997-2019.csv`) and use `sample_n` for large files. Or use path `data/sample_longitudinal.csv` (in repo) for a small demo CSV.

## Approach

1. **Context-aware detection:** Compare each person to their **own** baseline (mean of early waves), not population. *Life events* (job loss, retirement, divorce, stress): NLSY97 uses a 6th longitudinal variable as `life_event_proxy`; Psycho-emotional follow-up uses a life-events template; config `LIFE_EVENT_COLS` for employment/marital when mapped.
2. **Weak signals:** Moving average change, z-score vs personal history, trend slope; flag declining trends.
3. **Risk score:** 0–100 from a lightweight model (e.g. Logistic Regression), F2-optimized.
4. **Risk category:** Psycho-emotional / Metabolic / Cardiovascular from dominant signals.
5. **Explainability:** Which features changed and by how much.
6. **Follow-up:** One supportive, non-diagnostic question (templates by signal type).

## Architecture (high level)

```
Data (CSV) → Load & align → Per-person baseline → Weak signals (slope, deviation, flags)
    → No-leakage target (last wave) + features (past only) → Train model (F2-optimized)
    → Risk score 0–100 + category → Explainability → Empathetic follow-up question
```

## No data leakage & feature audit

**Features used:** Baseline deviations, % change, z-scores, trend slopes, and declining flags only. No medication, diagnosis, or outcome-revealing variables. Target is defined from last wave only; model is trained on past waves only.

## Fairness

We do not use age, gender, or ethnicity as model inputs. When the dataset includes demographics, a fairness audit (e.g. stratified F2 or error rates by group) can be run; we keep the model simple and avoid demographic-based scoring.

**Could the model become biased?** Not by design — we never feed demographics into the model, so scores and categories are not directly based on age, gender, or ethnicity. Bias is still *possible*: e.g. if the training data has different rates of decline by group, the model can reflect that; selection bias (who is in the sample) can also matter. The follow-up question is chosen only by **top contributing feature** (e.g. health rating, activity, stress), not by demographics. To check for unfairness, run stratified metrics (F2, error rates) by group when demographics are available.

## Dataset

- **NLSY97:** When path contains `nlsy97` (e.g. `nlsy97_all_1997-2019.csv`), the loader uses **real NLSY97** data: reads first 51 columns (1 ID + 10 waves × 5 vars), reshapes wide→long (one row per person per wave). Max rows = number of **persons** (up to 8984).
- **Sample CSV** in repo: `data/sample_longitudinal.csv` (synthetic, same structure).
- No **data leakage** — do not use features that directly reveal the outcome (e.g. medication for the predicted condition).

## License & compliance

Open-source; ≥80% public code. No diagnosis or treatment recommendations.

---

Hea Hackathon · AltaIR Capital, Harbour.Space, Nebius
