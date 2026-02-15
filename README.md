# Hea — Personal Health Drift Detector (PHDD)

Longitudinal health risk detection from self-reported data. Detects **personal baseline drift** and **weak signals**, outputs a **risk score (0–100)**, **risk category**, **explanation**, and an **empathetic follow-up question**. No diagnosis, no treatment advice.

**Live app:** [https://hea-hackathon.streamlit.app/](https://hea-hackathon.streamlit.app/)

---

## Problem & Vision

### Problems this solution solves

| Problem | How PHDD addresses it |
|--------|------------------------|
| **Late detection** | Flags **early** changes by comparing each person to their **own** baseline over time, not to population norms — so drift is visible before it becomes severe. |
| **One-size-fits-all** | Uses **personal baselines** and **weak signals** (slope, deviation, trends) so what “normal” means is per person, not averaged across everyone. |
| **Black-box risk** | Provides a **risk score (0–100)**, **category** (Psycho-emotional / Metabolic / Cardiovascular), and **explanation** of which factors changed and by how much. |
| **Cold or clinical tone** | Adds an **empathetic follow-up question** (no diagnosis, no treatment) so the output supports conversation rather than alarming the user. |
| **Life events ignored** | Incorporates a **life-events proxy** (e.g. job loss, retirement, divorce, stress) in signals and follow-up so context like major life changes is reflected. |
| **Data leakage / unfair comparisons** | Uses a **no-leakage** setup: target from last wave only, features from past waves only; optional **fairness** checks by demographic group without using demographics as inputs. |

### Vision

**Personal, early, explainable, and supportive.**  
We aim for a world where health risk is understood as **change from your own normal** — not from a population average — and where early signals are surfaced in a way that is **interpretable** and **conversation-friendly**, without diagnosing or prescribing. PHDD is a step toward that: longitudinal self-reported data → personal baseline + weak signals → risk score + category + “why we flagged” + one empathetic question, so people and care teams can **talk about what’s changing** before it becomes a crisis.

---

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
│   ├── data_loader.py     # Load longitudinal CSV, NLSY97 wide→long, handle missing
│   ├── baseline.py        # Per-person baseline builder
│   ├── weak_signals.py    # Moving average, trend slope, flags
│   ├── target_no_leakage.py  # Target last wave, features past only
│   ├── risk_model.py      # Score 0–100, risk category, F2-optimized
│   ├── explainability.py  # Feature importance, human-readable summary
│   ├── follow_up.py       # Empathetic question templates
│   ├── fairness.py        # Stratified metrics by group (optional)
│   └── pipeline.py        # End-to-end pipeline
├── data/
│   └── sample_longitudinal.csv   # Sample (in repo) — demo without NLSY97
├── DATA_NLSY97.md         # NLSY97 data description
├── HACKATHON_CHECKLIST.md # Submission checklist vs judging criteria
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

**App features:** Filter results by risk band (Low/Moderate/High) and category; **download full results (last wave) as CSV**; **feature importance** (top factors driving the model) in an expander; **choose a person by ID** to see their explanation and follow-up question.

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

## Data description (tables)

### Dataset overview

| Dataset | File | Source | Format | Loader | Max rows |
|--------|------|--------|--------|--------|----------|
| **NLSY97** | `nlsy97_all_1997-2019.csv` | BLS/NLS (National Longitudinal Survey of Youth 1997) | Wide (1 row per person) | Path contains `nlsy97` → read first 61 cols, reshape wide→long | 8,984 persons |
| **Sample** | `data/sample_longitudinal.csv` | Repo (synthetic) | Long (person–wave rows) | Default when path not NLSY97 | 2,000 (configurable) |

### NLSY97 file structure (as used in this project)

| Item | Value |
|------|--------|
| **Columns read** | First **61** (column 0 = person ID + 10 waves × 6 variables) |
| **Reshape** | Wide → long: one row per **person–wave**; cols: `PUBID`, `wave` (0..9), 6 feature cols |
| **Person ID** | Column 0 → `PUBID` (or row index if ID missing) |
| **Cohort** | ~8,984 respondents (born 1980–1984, ages 12–16 as of Dec 31, 1996) |
| **Waves in file** | First 10 waves used (0..9); NLSY97 has more rounds in full survey |

### NLSY97 variables (logical names)

| Position per wave | Logical name | Role in pipeline |
|------------------|---------------|------------------|
| 1 | `health_rating` | Self-reported health (baseline + signals) |
| 2 | `stress_level` | Stress / mental load |
| 3 | `activity_level` | Physical activity |
| 4 | `var_3` | Additional health/lifestyle |
| 5 | `var_4` | Additional health/lifestyle |
| 6 | `life_event_proxy` | Proxy for life events (job loss, retirement, divorce, stress) |

### NLSY97 missing codes (converted to NaN)

| Code | Meaning (typical NLSY97) |
|------|---------------------------|
| -5 | Refusal / invalid |
| -4 | Non-interview |
| -3 | Skip |
| -2 | Out of universe |
| -1 | Other non-valid |

### Sample CSV (`data/sample_longitudinal.csv`)

| Item | Value |
|------|--------|
| **Format** | Long: `PUBID`, `wave`, plus health/lifestyle columns (e.g. `health_rating`, `stress_level`, `activity_level`) |
| **Purpose** | Demo without NLSY97 file; same structure as pipeline expects after NLSY97 reshape |
| **Rows** | ~30k (configurable); first 50 cols read for non-NLSY97 paths |

### Data description references

| Link | Description |
|------|-------------|
| [NLSY97 cohort index](https://www.nlsinfo.org/content/cohorts/nlsy97) | NLS Info — NLSY97 cohort |
| [NLSY97 documentation](https://nlsinfo.org/content/cohorts/nlsy97/using-and-understanding-the-data/nlsy97-documentation) | Using and understanding NLSY97 data |
| [BLS NLSY97 overview](https://www.bls.gov/nls/nlsy97.htm) | Bureau of Labor Statistics NLSY97 |

**No data leakage:** We do not use features that directly reveal the outcome (e.g. medication for the predicted condition). Target = last wave only; features = past waves only.

## License & compliance

Open-source; ≥80% public code. No diagnosis or treatment recommendations.

---

Hea Hackathon · AltaIR Capital, Harbour.Space, Nebius
