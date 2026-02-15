# Full project check

Last check: project structure, code, config, docs, and minor fixes.

---

## ✅ Structure

| Item | Status |
|------|--------|
| **Root** | `app.py`, `requirements.txt`, `README.md`, `.gitignore` |
| **src/** | `config`, `data_loader`, `baseline`, `weak_signals`, `target_no_leakage`, `risk_model`, `explainability`, `follow_up`, `fairness`, `pipeline` |
| **data/** | `sample_longitudinal.csv` (in repo); NLSY97 CSV ignored (large) |
| **Docs** | `README.md`, `DATA_NLSY97.md`, `RULES_GAP.md`, `HACKATHON_CHECKLIST.md`, `TASKS_AND_SUBTASKS.md`, `NEXT_STEPS.md` |
| **notebooks/** | Optional (may be empty) |

---

## ✅ Dependencies

| Package | Version | Use |
|---------|---------|-----|
| pandas | ≥1.5.0 | Data load, pipeline |
| numpy | ≥1.23.0 | Numerics |
| scikit-learn | ≥1.2.0 | Model, metrics, split |
| streamlit | ≥1.28.0 | App |

---

## ✅ Code flow

1. **Load** (`data_loader`): CSV line-by-line; NLSY97 → wide→long, 61 cols, 6 vars, missing -5..-1 → NaN.
2. **Baseline** (`baseline`): Per-person mean/std from early waves.
3. **Weak signals** (`weak_signals`): Moving average change, trend slope, declining flags.
4. **No leakage** (`target_no_leakage`): Target = last wave; features = past waves only.
5. **Model** (`risk_model`): LogisticRegression, F2-optimized, StandardScaler.
6. **Fairness** (`fairness`): Optional stratified metrics when `DEMOGRAPHIC_COLS` set.
7. **Explain** (`explainability`): Top contributors, human-readable changes, main_change_names.
8. **Follow-up** (`follow_up`): Template by main change / category; life-events for Psycho-emotional.

---

## ✅ Config

| Config | Purpose |
|--------|---------|
| `ID_COL`, `WAVE_COL` | Longitudinal keys |
| `HEALTH_LIFESTYLE_COLS` | Feature names (health_rating, stress_level, etc.) |
| `LIFE_EVENT_COLS` | Optional life-event columns when mapped |
| `DEMOGRAPHIC_COLS` | Optional; fairness stratification only (not model inputs) |
| `RANDOM_STATE` | Reproducibility |

---

## ✅ Lint

- No linter errors on `app.py`, `src/pipeline.py`, `src/config.py`, `src/data_loader.py`, `src/fairness.py`, `src/follow_up.py`.

---

## Fixes applied in this check

1. **app.py:** Help text "10 waves × 5 vars" → "10 waves × 6 vars".
2. **pipeline.py:** Removed redundant `fairness_result = None` before the fairness block.
3. **README.md:** Repo structure updated to include `target_no_leakage.py`, `fairness.py`, `DATA_NLSY97.md`.

---

## Optional next steps

- Create `notebooks/` with a single placeholder or demo notebook if you want it in the tree.
- Add a minimal test (e.g. `tests/test_pipeline.py`) that runs `run_pipeline()` on synthetic data and checks keys in the result.
- When NLSY97 has demographic columns, set `DEMOGRAPHIC_COLS` in `config.py` and re-run to see fairness expander in the app.
