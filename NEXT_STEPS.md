# Next Steps — Git, Deploy, and Hackathon Checklist

## 1. Git push (get code on GitHub)

Your repo: **https://github.com/SifatAli008/Hea-Hackathon**

**If the folder is not yet a git repo:**

```bash
cd e:\Hea
git init
git remote add origin https://github.com/SifatAli008/Hea-Hackathon.git
```

**Add and push (avoid pushing huge data files):**

```bash
# Create .gitignore so you don't push 7GB CSV or secrets
# Then:
git add .
git status   # check nothing huge is staged
git commit -m "PHDD prototype: pipeline, Streamlit app, README"
git branch -M main
git push -u origin main
```

**If the repo already exists and you have other branches:**

```bash
git add .
git commit -m "Add PHDD prototype"
git push origin main
```

---

## 2. .gitignore (do this before first push)

So you don’t push data files, venv, or secrets:

```
# Data (large files)
*.csv
*.dat
*.sas
*.sdf
*.cdb
*.R
.NLSY97
data/

# Python
__pycache__/
*.py[cod]
.venv/
venv/
env/
*.egg-info/

# IDE / OS
.idea/
.vscode/
.DS_Store
Thumbs.db

# Secrets (if you ever add env files)
.env
.env.local
*.pem
```

---

## 3. “Deploy” / run on Nebius (for judging)

Judges will use **Nebius**, not your laptop. So “deploy” = **run the app on the Nebius instance**.

**On the Nebius Jupyter instance (http://89.169.121.184:8888):**

1. **Clone the repo** (after you push):
   ```bash
   cd /path/you/work
   git clone https://github.com/SifatAli008/Hea-Hackathon.git
   cd Hea-Hackathon
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Data path:**  
   - If NLSY97 is in the S3 bucket mounted at `/s3`, set:
     ```bash
     export HEA_DATA_PATH=/s3/hackathon-team-fabric3-6
     ```
   - Or put the CSV path in the app when you run it (“Use path (NLSY97 or /s3/...)”).

4. **Run Streamlit:**
   ```bash
   python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   ```
   Then the app is available at `http://89.169.121.184:8501` (if the instance allows it).

5. **Or run in Jupyter:**  
   Create a notebook that runs `run_pipeline()` and shows metrics + sample outputs (same as the app flow). Judges can run the notebook if they prefer.

---

## 4. Anything else (hackathon checklist)

| Task | Status / action |
|------|------------------|
| **GitHub repo** | Push code, README, setup instructions. ≥80% open source. |
| **README** | Already in repo. Ensure it has: problem, dataset, approach, how to run on Nebius. |
| **Run on real data** | On Nebius, point to NLSY97 in `/s3`, run pipeline, record F2 / PR-AUC / ROC-AUC. |
| **Fairness checks** | In code or notebook: stratify metrics by age, gender, ethnicity (if in data). Document in README. |
| **No data leakage** | Audit feature list; document that no disease-revealing features are used. |
| **5-min presentation** | Problem → Dataset → Approach → Demo → Impact. Rehearse with the Nebius app or notebook. |
| **Demo flow** | Data → baseline → weak signals → risk score → explanation → follow-up question. |

---

## 5. One-line summary

**Next:**  
1) Add `.gitignore`, then **git push** to `SifatAli008/Hea-Hackathon`.  
2) On **Nebius**, clone repo, `pip install -r requirements.txt`, set `HEA_DATA_PATH` if needed, run **Streamlit** (or a notebook).  
3) Run on **real NLSY97** from S3, add **fairness** and **no-leakage** notes, and prepare the **5-min presentation**.
