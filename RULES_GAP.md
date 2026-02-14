# Rules.pdf — What’s Missing

Checked against **Rules.pdf** (solution requirements + what you must submit).

---

## ✅ Fully covered by your submission

| Rules requirement | Your status |
|-------------------|-------------|
| **Context-aware detection** | ✓ Compare to own baseline, detect changes over time, weak signals |
| **Risk score** | ✓ Numeric 0–100 |
| **Risk category** | ✓ Cardiovascular, Metabolic, Psycho-emotional |
| **Safe & empathetic follow-up** | ✓ Supportive question, no diagnosis, no treatment |
| **Working prototype** | ✓ Streamlit app (data processing, anomaly detection, risk scoring, follow-up) |
| **GitHub** | ✓ Source code, README, setup, 80%+ open source |
| **No diagnosing** | ✓ No disease declaration, no treatment recommendation |
| **Open source 80%+** | ✓ Met |

---

## ⚠️ Missing or partial vs Rules.pdf

### **Life events — filled**

**Rules say:**  
*“Consider major life events (e.g., job loss, retirement, divorce, stress).”*

**Filled:** NLSY97 uses 6th longitudinal variable as `life_event_proxy`; Psycho-emotional follow-up uses life-events template; config has `LIFE_EVENT_COLS` for employment/marital when mapped.

*(Obsolete — gap filled.)*


---

### **Short team presentation (5 min) — team action**

**Rules say:**  
*“Short Team Presentation — Brief pitch format (5 minutes): 1. Problem 2. Dataset used 3. Approach 4. Demo 5. Impact.”*

**Current state:**  
This is **not** in the repo; the team must **prepare and deliver** it live.

**What to do:**  
Prepare the 5-min pitch (Problem → Dataset → Approach → Demo → Impact) and rehearse with the live app (e.g. https://hea-hackathon.streamlit.app/).

---

## Summary

- **Life events gap filled:** NLSY97 uses 6th var as `life_event_proxy`; follow-up has life-events template for Psycho-emotional; config has `LIFE_EVENT_COLS` for future mapping.
- **Follow-up:** Not fixed — varies by top contributors and category; reproducible variety via `random.choice(templates)` seeded by risk_score.
- **Only remaining Rules item:** 5-min presentation (live; team prepares and delivers).
