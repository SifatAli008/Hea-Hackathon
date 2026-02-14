# Hea — Personal Health Drift Detector (PHDD)

Longitudinal health risk detection from self-reported data. Detects **personal baseline drift** and **weak signals**, outputs a **risk score (0–100)**, **risk category**, **explanation**, and an **empathetic follow-up question**. No diagnosis, no treatment advice.

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
└── notebooks/             # Optional exploration
```

## Run demo

**Streamlit (recommended):**

```bash
streamlit run app.py
```

**Synthetic demo (no data file):**  
Use “Synthetic demo (no file)” in the sidebar — runs on small generated data.

**With data:**  
- Upload a CSV in the app, or  
- Set “Use path” to your NLSY97 path (e.g. `/s3/.../nlsy97_all_1997-2019.csv`) and use `sample_n` for large files. Or use path `data/sample_longitudinal.csv` (in repo) for a small demo CSV.

## Approach

1. **Context-aware detection:** Compare each person to their **own** baseline (mean of early waves), not population.
2. **Weak signals:** Moving average change, z-score vs personal history, trend slope; flag declining trends.
3. **Risk score:** 0–100 from a lightweight model (e.g. Logistic Regression), F2-optimized.
4. **Risk category:** Psycho-emotional / Metabolic / Cardiovascular from dominant signals.
5. **Explainability:** Which features changed and by how much.
6. **Follow-up:** One supportive, non-diagnostic question (templates by signal type).

## Dataset

- **NLSY97** (or other longitudinal self-reported health data).  
- Important: no **data leakage** — do not use features that directly reveal the outcome (e.g. medication for the predicted condition).

## License & compliance

Open-source; ≥80% public code. No diagnosis or treatment recommendations.

---

Hea Hackathon · AltaIR Capital, Harbour.Space, Nebius
