# Data description: `nlsy97_all_1997-2019.csv`

## Source

**National Longitudinal Survey of Youth 1997 (NLSY97)**  
- Bureau of Labor Statistics (BLS) / National Longitudinal Surveys (NLS).  
- Documentation: [NLSY97 at NLS Info](https://www.nlsinfo.org/content/cohorts/nlsy97), [BLS NLSY97](https://www.bls.gov/nls/nlsy97.htm).

## Cohort

- **Sample:** ~8,984 respondents (cohort born 1980–1984, ages 12–16 as of December 31, 1996).  
- **Waves:** Survey conducted annually (1997–2011) and biennially since; multiple rounds (Round 1 through Round 20+).  
- **Topics:** Education, employment, family formation, health-related and lifestyle items, geography, program participation.

## File structure (as used in this project)

- **Format:** Wide — one row per person; columns are typically one per variable per wave (or per round).  
- **Our loader:** Reads the **first 61 columns** (1 person ID + 10 waves × 6 variables).  
- **Reshape:** Wide → long: one row per **person–wave**; columns: `PUBID`, `wave` (0..9), plus 6 feature columns.  
- **Max persons:** Up to 8,984 rows (persons) in the CSV; “max rows” in the app = number of persons to load.

## Variables (logical names in this repo)

The loader maps the first 6 variables per wave to these names (actual NLSY97 variable codes depend on the CSV extract and codebook):

| Position | Logical name        | Role in pipeline                          |
|----------|---------------------|-------------------------------------------|
| 1        | `health_rating`     | Self-reported health (baseline + signals) |
| 2        | `stress_level`      | Stress / mental load                      |
| 3        | `activity_level`    | Physical activity                         |
| 4        | `var_3`             | Additional health/lifestyle               |
| 5        | `var_4`             | Additional health/lifestyle               |
| 6        | `life_event_proxy`  | Proxy for life events (Rules: job loss, retirement, divorce, stress) |

To use actual NLSY97 names, set the corresponding codes in `src/config.py` (e.g. `HEALTH_LIFESTYLE_COLS`, `LIFE_EVENT_COLS`) and align the CSV column order with the loader’s expected 6 variables per wave.

## Missing and invalid codes

NLSY97 uses negative codes for non-valid responses. The loader converts these to `NaN`:

- **-5, -4, -3, -2, -1** → `NaN` (e.g. refusals, skip, out-of-universe, non-interview).

Other negative or non-numeric values in the CSV may remain; the pipeline uses per-person forward/backward fill and column-level missing fraction (e.g. `max_missing_frac=0.95` for NLSY97) before modeling.

## Person ID

- **CSV:** Column 0 is treated as person ID and renamed to `PUBID`.  
- If the file uses NLSY97 reference IDs (e.g. masked PUBID), the loader can use that; otherwise row index (0..N-1) is used as person ID for the wide→long reshape.

## References

- [NLSY97 cohort index](https://www.nlsinfo.org/content/cohorts/nlsy97)  
- [NLSY97 documentation](https://nlsinfo.org/content/cohorts/nlsy97/using-and-understanding-the-data/nlsy97-documentation)  
- [BLS NLSY97 overview](https://www.bls.gov/nls/nlsy97.htm)
