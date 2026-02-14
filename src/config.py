"""
Configuration: paths, column names, and constants.
Data can live under /s3 on Nebius (S3 bucket) or locally.
"""
import os
from pathlib import Path

# Base path: use /s3 when on Nebius, else current dir or env
DATA_BASE = os.environ.get("HEA_DATA_PATH", ".")
DATA_PATH = Path(DATA_BASE)

# NLSY97: adjust to your file location (e.g. DATA_PATH / "nlsy97_all_1997-2019.csv")
# On Nebius: e.g. Path("/s3/hackathon-team-fabric3-6/nlsy97_all_1997-2019.csv")
NLSY97_CSV = DATA_PATH / "nlsy97_all_1997-2019.csv"

# Column names - NLSY97 uses codes (B0000200 etc). Map to logical names here if you have a codebook.
# Otherwise we'll infer from first row or use these as expected names after renaming.
ID_COL = "PUBID"  # or actual code e.g. "B0000200" for person ID
WAVE_COL = "wave"  # or year/round column name
TIME_COL = "year"  # numeric year or wave index

# Features used for baseline + weak signals (no leakage).
# Replace with actual NLSY97 variable names/codes from your codebook.
HEALTH_LIFESTYLE_COLS = [
    "health_rating",   # self-reported health 1-5
    "sleep_hours",     # if available
    "activity_level",  # if available
    "stress_level",    # if available
    "bmi",             # if available
]
# If using raw NLSY97 codes, set HEALTH_LIFESTYLE_COLS to list of actual column names.

# Risk score bands
RISK_LOW = (0, 30)
RISK_MODERATE = (31, 60)
RISK_HIGH = (61, 100)

# Random seed for reproducibility
RANDOM_STATE = 42
