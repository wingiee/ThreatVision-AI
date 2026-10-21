import os
import numpy as np
import joblib
import pandas as pd
import json
from datetime import datetime

# ==========================================
# PATHS
# ==========================================
OUTPUT_DIR = r"C:\Users\Archi\OneDrive\Desktop\IntrusionDetection\processed"
MODEL_DIR = r"C:\Users\Archi\OneDrive\Desktop\IntrusionDetection\models"
REPORT_DIR = r"C:\Users\Archi\OneDrive\Desktop\IntrusionDetection\reports"

os.makedirs(REPORT_DIR, exist_ok=True)

# ==========================================
# LOAD PREDICTIONS + LABEL MAP
# ==========================================
print("=" * 60)
print("Loading predictions and label map...")
print("=" * 60)

y_test = np.load(os.path.join(OUTPUT_DIR, "y_test.npy"))
y_pred = np.load(os.path.join(OUTPUT_DIR, "y_pred.npy"))
y_proba = np.load(os.path.join(OUTPUT_DIR, "y_proba.npy"))
label_map = joblib.load(os.path.join(MODEL_DIR, "label_map.pkl"))

# Guaranteed ordered class names
class_names = [label_map[i] for i in range(len(label_map))]
print(f"Classes : {class_names}")

# ==========================================
# BASE SEVERITY TABLE
# Attack type → base severity score (0–10)
# Designed based on real-world impact
# ==========================================
SEVERITY_TABLE = {
    "BENIGN" : 0,
    "benign" : 0,
    "Bot Attack" : 8,
    "DDoS" : 9,
    "DoS GoldenEye" : 7,
    "DoS Hulk" : 7,
    "DoS Slowhttptest" : 6,
    "DoS slowloris" : 6,
    "FTP-Patator" : 6,
    "SSH-Patator" : 6,
    "Infiltration" : 10,
    "Port Scan" : 3,
    "Web Attack - Brute Force" : 6,
    "Web Attack - Sql Injection": 8,
    "Web Attack - XSS" : 5,
}

# ==========================================
# THREAT LEVEL MAPPING
# Score 0–10 → Low / Medium / High / Critical
# ==========================================
def get_threat_level(score):
    if score == 0:
        return "None"
    elif score < 3:
        return "Low"
    elif score < 6:
        return "Medium"
    elif score < 8.5:
        return "High"
    else:
        return "Critical"

# ==========================================
# RISK SCORE CALCULATOR
# Formula: Risk Score = Base Severity × Confidence
# ==========================================
def calculate_risk_score(attack_name, confidence):
    base = SEVERITY_TABLE.get(attack_name, 5) # default 5 if unknown
    risk = round(base * confidence, 2)
    return risk

# ==========================================
# PROCESS ALL TEST PREDICTIONS
# ==========================================
print("\nCalculating Risk Scores for all test samples...\n")

records = []

for i in range(len(y_pred)):
    pred_class_idx = y_pred[i]
    true_class_idx = y_test[i]

    pred_label = label_map[pred_class_idx]
    true_label = label_map[true_class_idx]

    # Confidence = probability of the predicted class
    confidence = float(y_proba[i][pred_class_idx])

    # Risk Score
    risk_score = calculate_risk_score(pred_label, confidence)

    # Threat Level
    threat_level = get_threat_level(risk_score)

    records.append({
        "sample_id" : i,
        "true_label" : true_label,
        "pred_label" : pred_label,
        "confidence" : round(confidence, 4),
        "base_severity": SEVERITY_TABLE.get(pred_label, 5),
        "risk_score" : risk_score,
        "threat_level" : threat_level,
        "correct" : pred_label == true_label
    })

# ==========================================
# BUILD DATAFRAME
# ==========================================
df = pd.DataFrame(records)

print(df[["pred_label", "confidence", "risk_score", "threat_level"]].head(20).to_string())

# ==========================================
# SUMMARY STATISTICS
# ==========================================
print("\n" + "=" * 60)
print("RISK SCORE SUMMARY")
print("=" * 60)

print(f"\nTotal Samples : {len(df)}")
print(f"Benign Traffic : {(df['pred_label'].isin(['BENIGN', 'Benign'])).sum()}")
print(f"Threats Detected : {(df['pred_label'] != 'Benign').sum()}")

print("\nThreat Level Distribution:")
level_counts = df['threat_level'].value_counts()
for level, count in level_counts.items():
    pct = count / len(df) * 100
    print(f" {level:<12} : {count:>8,} ({pct:.2f}%)")

print("\nAverage Risk Score by Attack Type:")
avg_risk = (
    df[df['pred_label'] != 'Benign']
    .groupby('pred_label')['risk_score']
    .mean()
    .sort_values(ascending=False)
)
for attack, score in avg_risk.items():
    level = get_threat_level(score)
    print(f" {attack:<35} : {score:.2f} [{level}]")

# ==========================================
# SAVE FULL RESULTS TO CSV
# ==========================================
csv_path = os.path.join(REPORT_DIR, "risk_scores.csv")
df.to_csv(csv_path, index=False)
print(f"\nFull risk score table saved: {csv_path}")

# ==========================================
# SAVE SUMMARY JSON (used by dashboard later)
# ==========================================
summary = {
    "generated_at" : datetime.now().isoformat(),
    "total_samples" : len(df),
    "benign_count" : int((df['pred_label'] == 'Benign').sum()),
    "threat_count" : int((df['pred_label'] != 'Benign').sum()),
    "threat_levels" : level_counts.to_dict(),
    "avg_risk_by_attack" : avg_risk.round(2).to_dict(),
    "severity_table" : SEVERITY_TABLE,
}
json_path = os.path.join(REPORT_DIR, "risk_summary.json")
with open(json_path, "w") as f:
    json.dump(summary, f, indent=4)
print(f"Risk summary JSON saved : {json_path}")

# ==========================================
# SAMPLE OUTPUT — show 5 high-risk detections
# ==========================================
print("\n" + "=" * 60)
print("SAMPLE HIGH-RISK DETECTIONS")
print("=" * 60)

high_risk = df[df['threat_level'].isin(['High', 'Critical'])].head(5)

for _, row in high_risk.iterrows():
    print(f"""
  Sample ID : {row['sample_id']}
  Attack Type : {row['pred_label']}
  Confidence : {row['confidence'] * 100:.1f}%
  Base Severity : {row['base_severity']} / 10
  Risk Score : {row['risk_score']} / 10
  Threat Level : {row['threat_level']}
  Correct Pred : {row['correct']}
  {'-' * 40}""")

# ==========================================
# SAVE RISK MODULE (callable from dashboard)
# ==========================================
risk_module = {
    "severity_table" : SEVERITY_TABLE,
    "get_threat_level" : "score < 3 → Low | < 6 → Medium | < 8.5 → High | >= 8.5 → Critical",
    "formula" : "Risk Score = Base Severity × Model Confidence"
}
joblib.dump(
    {"severity_table": SEVERITY_TABLE},
    os.path.join(MODEL_DIR, "risk_module.pkl")
)
print(f"\nRisk module config saved: {MODEL_DIR}\\risk_module.pkl")