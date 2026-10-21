"""
isolation_forest.py
-------------------
Unknown attack detection using Isolation Forest.

Runs AFTER XGBoost. Strategy:
  - XGBoost already classified samples into 10 known classes.
  - Isolation Forest is trained on BENIGN traffic only (learns the normal profile).
  - At inference it scores every sample; those that are "too far" from the normal
    profile AND were predicted as an attack get flagged as potential unknown threats.
  - Results are merged into risk_scores.csv as two new columns:
      anomaly_score   : raw IF score  (more negative = more anomalous)
      unknown_threat  : True / False  (anomaly AND not Benign predicted)

Output:
  reports/risk_scores_with_anomaly.csv   — enriched risk table
  reports/anomaly_summary.json           — summary stats
  models/isolation_forest.pkl            — trained IF model (reusable in dashboard)
"""

import os
import json
import numpy as np
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

# ──────────────────────────────────────────────
# PATHS  (relative to this file — works on any machine)
# ──────────────────────────────────────────────
# BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = r"C:\Users\Archi\OneDrive\Desktop\IntrusionDetection\processed"
MODEL_DIR  = r"C:\Users\Archi\OneDrive\Desktop\IntrusionDetection\models"
REPORT_DIR = r"C:\Users\Archi\OneDrive\Desktop\IntrusionDetection\reports"
# REPORT_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
IF_CONTAMINATION = 0.02   # Expected fraction of anomalies in full test set.
                           # Tune this: lower → fewer flagged, higher → more flagged.
IF_N_ESTIMATORS  = 200    # More trees = more stable scores (slower to train).
IF_RANDOM_STATE  = 42
CONFIDENCE_THRESHOLD = 0.6  # Only flag IF anomaly when XGBoost confidence < this
                              # for Benign, i.e. model itself was uncertain it was normal.


# ──────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────
print("=" * 60)
print("ISOLATION FOREST — Unknown Attack Detection")
print("=" * 60)

print("\n[1] Loading data...")

X_train = np.load(os.path.join(OUTPUT_DIR, "X_train.npy"))
X_test = np.load(os.path.join(OUTPUT_DIR, "X_test.npy"))
y_train = np.load(os.path.join(OUTPUT_DIR, "y_train.npy"))
y_test = np.load(os.path.join(OUTPUT_DIR, "y_test.npy"))
y_pred = np.load(os.path.join(OUTPUT_DIR, "y_pred.npy"))
y_proba = np.load(os.path.join(OUTPUT_DIR, "y_proba.npy"))

label_map = joblib.load(
    os.path.join(MODEL_DIR, "label_map.pkl")
)

feature_names = joblib.load(
    os.path.join(MODEL_DIR, "feature_names.pkl")
)

# Reverse map: class name → index (used to find Benign index)
name_to_idx = {v: k for k, v in label_map.items()}
BENIGN_IDX  = name_to_idx.get("Benign", 0)

print(f"   X_train : {X_train.shape}")
print(f"   X_test  : {X_test.shape}")
print(f"   Benign class index : {BENIGN_IDX}")
print(f"   Classes : {list(label_map.values())}")


# ──────────────────────────────────────────────
# 2. BUILD BENIGN-ONLY TRAINING SET
#    Isolation Forest learns what "normal" looks like.
#    We ONLY train it on Benign samples so any deviation = anomaly.
# ──────────────────────────────────────────────
print("\n[2] Extracting Benign training samples...")

benign_mask_train = (y_train == BENIGN_IDX)
X_train_benign    = X_train[benign_mask_train]

print(f"   Benign training samples : {X_train_benign.shape[0]:,}")
print(f"   (out of {X_train.shape[0]:,} total training samples)")


# ──────────────────────────────────────────────
# 3. TRAIN ISOLATION FOREST
# ──────────────────────────────────────────────
print("\n[3] Training Isolation Forest...")
print(f"   contamination={IF_CONTAMINATION}, n_estimators={IF_N_ESTIMATORS}")

iso_forest = IsolationForest(
    n_estimators  = IF_N_ESTIMATORS,
    contamination = IF_CONTAMINATION,
    random_state  = IF_RANDOM_STATE,
    n_jobs        = -1,          # use all CPU cores
    verbose       = 0
)
iso_forest.fit(X_train_benign)

print("   Training complete.")

# Save the model for reuse in the Streamlit dashboard
if_model_path = os.path.join(
    MODEL_DIR,
    "isolation_forest.pkl"
)
joblib.dump(iso_forest, if_model_path)
print(f"   Model saved → {if_model_path}")


# ──────────────────────────────────────────────
# 4. SCORE THE TEST SET
#    score_samples() returns negative avg path length.
#    More negative = more anomalous.
#    predict() returns +1 (normal) or -1 (anomaly).
# ──────────────────────────────────────────────
print("\n[4] Scoring test set...")

if_scores    = iso_forest.score_samples(X_test)   # shape: (N,)  raw anomaly scores
if_labels    = iso_forest.predict(X_test)          # shape: (N,)  +1 or -1

# Normalise scores to [0, 1] so they're dashboard-friendly
#   0 = most normal,  1 = most anomalous
score_min, score_max = if_scores.min(), if_scores.max()
anomaly_score_norm = 1 - (if_scores - score_min) / (score_max - score_min + 1e-9)

print(f"   Samples flagged as anomalies by IF   : {(if_labels == -1).sum():,}")
print(f"   Samples labelled as normal by IF     : {(if_labels == 1).sum():,}")


# ──────────────────────────────────────────────
# 5. UNKNOWN THREAT LOGIC
#
#    A sample is an "unknown threat" when ALL of:
#      (a) XGBoost predicted it as a known attack (not Benign)
#      (b) Isolation Forest flagged it as anomalous (-1)
#      (c) XGBoost's confidence for its predicted class is below threshold
#          OR the sample sits far from any cluster the model knows
#
#    The dual-gate (XGBoost + IF) keeps false positives low.
# ──────────────────────────────────────────────
print("\n[5] Applying unknown threat logic...")

xgb_predicted_attack = (y_pred != BENIGN_IDX)           # XGB says it's an attack
if_anomaly            = (if_labels == -1)                # IF says it's anomalous
xgb_low_confidence   = (y_proba.max(axis=1) < CONFIDENCE_THRESHOLD)

# Core flag: anomalous AND predicted as attack
unknown_threat = (
    (y_pred == BENIGN_IDX) &
    if_anomaly
)

# Bonus column: anomalous AND (attack OR low-confidence Benign)
# This catches Benign predictions that the model wasn't sure about
suspicious = if_anomaly & (xgb_predicted_attack | xgb_low_confidence)

print(f"   XGBoost predicted attack  : {xgb_predicted_attack.sum():,}")
print(f"   IF flagged anomalous       : {if_anomaly.sum():,}")
print(f"   → Unknown threats (both)  : {unknown_threat.sum():,}")
print(f"   → Suspicious (incl. low-conf Benign) : {suspicious.sum():,}")


# ──────────────────────────────────────────────
# 6. MERGE INTO risk_scores.csv
# ──────────────────────────────────────────────
print("\n[6] Merging into risk_scores.csv...")

risk_df = pd.read_csv(
    os.path.join(
        REPORT_DIR,
        "risk_scores.csv"
    )
)

risk_df["if_raw_score"]     = if_scores
risk_df["anomaly_score"]    = anomaly_score_norm.round(4)   # 0=normal, 1=anomalous
risk_df["if_label"]         = if_labels                     # +1 or -1
risk_df["unknown_threat"]   = unknown_threat
risk_df["suspicious"]       = suspicious

# Assign a threat category for display
def categorise(row):

    if row["unknown_threat"]:
        return "Unknown Threat"

    elif row["pred_label"] != "Benign":
        return "Known Attack"

    elif row["suspicious"]:
        return "Suspicious"

    else:
        return "Benign"

risk_df["detection_category"] = risk_df.apply(categorise, axis=1)

output_csv = os.path.join(
    REPORT_DIR,
    "risk_scores_with_anomaly.csv"
)
risk_df.to_csv(output_csv, index=False)
print(f"   Saved → {output_csv}")
print(f"   New columns: anomaly_score, if_label, unknown_threat, suspicious, detection_category")


# ──────────────────────────────────────────────
# 7. EVALUATION
#    Ground truth: anything not Benign should ideally be caught.
#    We treat the IF flag as a binary classifier and report metrics.
# ──────────────────────────────────────────────
print("\n[7] Evaluation...")
print("-" * 60)

# True binary labels: 0 = benign, 1 = attack
y_true_binary = (y_test != BENIGN_IDX).astype(int)
# IF prediction as binary: -1 → 1 (anomaly), +1 → 0 (normal)
y_if_binary   = (if_labels == -1).astype(int)

print("Isolation Forest — attack detection performance:")
print(classification_report(
    y_true_binary,
    y_if_binary,
    target_names=["Benign (normal)", "Attack (anomaly)"],
    digits=3
))

# Per-class breakdown: what fraction of each known attack did IF flag?
print("\nIF anomaly detection rate by XGBoost-predicted class:")
for idx, name in label_map.items():
    mask       = (y_pred == idx)
    if mask.sum() == 0:
        continue
    flagged    = if_anomaly[mask].sum()
    total      = mask.sum()
    rate       = flagged / total * 100
    marker     = " ◄ good" if (idx != BENIGN_IDX and rate > 30) else ""
    print(f"  {name:<40} : {flagged:>6,} / {total:>7,}  ({rate:5.1f}%){marker}")


# ──────────────────────────────────────────────
# 8. SUMMARY JSON
# ──────────────────────────────────────────────
print("\n[8] Saving summary...")

summary = {
    "generated_at"           : datetime.now().isoformat(),
    "model"                  : "IsolationForest",
    "config": {
        "contamination"      : IF_CONTAMINATION,
        "n_estimators"       : IF_N_ESTIMATORS,
        "trained_on"         : "Benign samples only",
        "benign_train_count" : int(X_train_benign.shape[0]),
    },
    "test_set": {
        "total_samples"      : int(len(if_labels)),
        "if_flagged_anomaly" : int((if_labels == -1).sum()),
        "if_flagged_normal"  : int((if_labels == 1).sum()),
    },
    "unknown_threat_detection": {
        "unknown_threats"    : int(unknown_threat.sum()),
        "suspicious"         : int(suspicious.sum()),
        "xgb_attack_samples" : int(xgb_predicted_attack.sum()),
        "logic"              : "XGBoost predicted attack AND IsolationForest flagged anomalous",
    },
    "per_class_anomaly_rate" : {},
}

for idx, name in label_map.items():
    mask = (y_pred == idx)
    if mask.sum() == 0:
        continue
    summary["per_class_anomaly_rate"][name] = {
        "total"        : int(mask.sum()),
        "flagged"      : int(if_anomaly[mask].sum()),
        "anomaly_rate" : round(float(if_anomaly[mask].mean()) * 100, 2),
    }

json_path = os.path.join(
    REPORT_DIR,
    "anomaly_summary.json"
)
with open(json_path, "w") as f:
    json.dump(summary, f, indent=4)
print(f"   Saved → {json_path}")


# ──────────────────────────────────────────────
# 9. SAMPLE OUTPUT — top unknown threats
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("SAMPLE UNKNOWN THREATS (top 8 by anomaly score)")
print("=" * 60)

unknown_df = risk_df[risk_df["unknown_threat"]].sort_values(
    "anomaly_score", ascending=False
).head(8)

cols = ["sample_id", "pred_label", "confidence", "risk_score",
        "threat_level", "anomaly_score"]
print(unknown_df[cols].to_string(index=False))

print("\n" + "=" * 60)
print("DONE.")
print(f"  Enriched CSV  → {output_csv}")
print(f"  Summary JSON  → {json_path}")
print(f"  IF model      → {if_model_path}")
print("=" * 60)