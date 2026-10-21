"""
=========================================================
ThreatVision AI
Prediction Engine
Part 1
---------------------------------------------------------
Loads models and preprocesses uploaded network traffic.
=========================================================
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =========================================================
# PATHS
# =========================================================

BASE_DIR = r"C:\Users\Archi\OneDrive\Desktop\IntrusionDetection"

MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

# =========================================================
# LOAD MODELS
# =========================================================

print("=" * 60)
print("Loading ThreatVision AI Models...")
print("=" * 60)

xgb_model = joblib.load(
    os.path.join(MODEL_DIR, "xgboost_model.pkl")
)

scaler = joblib.load(
    os.path.join(MODEL_DIR, "scaler.pkl")
)

label_encoder = joblib.load(
    os.path.join(MODEL_DIR, "label_encoder.pkl")
)

label_map = joblib.load(
    os.path.join(MODEL_DIR, "label_map.pkl")
)

feature_names = joblib.load(
    os.path.join(MODEL_DIR, "feature_names.pkl")
)

iso_model = joblib.load(
    os.path.join(MODEL_DIR, "isolation_forest.pkl")
)

print("✓ XGBoost Loaded")
print("✓ Isolation Forest Loaded")
print("✓ Scaler Loaded")
print("✓ Label Encoder Loaded")
print("✓ Feature Names Loaded")

# =========================================================
# LOAD RECOMMENDATIONS
# =========================================================

recommendation_df = pd.read_csv(
    os.path.join(
        REPORT_DIR,
        "recommendations.csv"
    )
)

print("✓ Recommendation Database Loaded")

# =========================================================
# LOAD RISK MODULE
# =========================================================

risk_module = joblib.load(
    os.path.join(
        MODEL_DIR,
        "risk_module.pkl"
    )
)

SEVERITY_TABLE = risk_module["severity_table"]

print("✓ Risk Module Loaded")

# =========================================================
# THREAT LEVEL
# =========================================================

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

# =========================================================
# RISK SCORE
# =========================================================

def calculate_risk_score(

        attack,

        confidence

):

    base = SEVERITY_TABLE.get(

        attack,

        5

    )

    return round(

        base * confidence,

        2

    )

# =========================================================
# LOAD CSV
# =========================================================

def load_csv(file):

    df = pd.read_csv(file)

    print("\nCSV Loaded Successfully")

    print(df.shape)

    return df

# =========================================================
# CLEAN COLUMN NAMES
# =========================================================

def clean_columns(df):

    df.columns = df.columns.str.strip()

    return df

# =========================================================
# HANDLE MISSING VALUES
# =========================================================

def handle_missing(df):

    df.replace(

        [np.inf, -np.inf],

        np.nan,

        inplace=True

    )

    df.fillna(

        0,

        inplace=True

    )

    return df

# =========================================================
# FEATURE CHECK
# =========================================================

def validate_features(df):

    missing = []

    for feature in feature_names:

        if feature not in df.columns:

            missing.append(feature)

            df[feature] = 0

    extra = [

        c

        for c in df.columns

        if c not in feature_names

    ]

    if len(extra):

        print(

            f"Ignoring {len(extra)} extra columns"

        )

    df = df[feature_names]

    return df

# =========================================================
# SCALE DATA
# =========================================================

def preprocess(df):

    df = clean_columns(df)

    df = handle_missing(df)

    df = validate_features(df)

    X = scaler.transform(df)

    return X

print("\nPrediction Engine Ready.")
print("=" * 60)

# =========================================================
# XGBOOST PREDICTION
# =========================================================

def predict_xgboost(X):

    predictions = xgb_model.predict(X)

    probabilities = xgb_model.predict_proba(X)

    confidence = probabilities.max(axis=1)

    labels = [
        label_map[int(i)]
        for i in predictions
    ]

    return predictions, labels, confidence, probabilities


# =========================================================
# ISOLATION FOREST
# =========================================================

def predict_anomaly(X):

    raw_scores = iso_model.score_samples(X)

    labels = iso_model.predict(X)

    score_min = raw_scores.min()
    score_max = raw_scores.max()

    anomaly_score = 1 - (
        (raw_scores - score_min)
        /
        (score_max - score_min + 1e-9)
    )

    return labels, anomaly_score


# =========================================================
# UNKNOWN THREAT LOGIC
# =========================================================

def detect_unknown_threat(

    attack_name,

    anomaly_label

):

    if anomaly_label == -1 and attack_name == "Benign":

        return True

    return False


# =========================================================
# DETECTION CATEGORY
# =========================================================

def detection_category(

    attack,

    unknown,

    anomaly_score

):

    if unknown:

        return "Unknown Threat"

    if attack != "Benign":

        return "Known Attack"

    if anomaly_score > 0.75:

        return "Suspicious"

    return "Benign"


# =========================================================
# BUILD PREDICTION TABLE
# =========================================================

def build_prediction_table(

        X,

        original_df

):

    pred_idx,\
    pred_label,\
    confidence,\
    probability = predict_xgboost(X)

    anomaly_label,\
    anomaly_score = predict_anomaly(X)

    rows = []

    for i in range(len(pred_label)):

        attack = pred_label[i]

        conf = float(confidence[i])

        risk = calculate_risk_score(

            attack,

            conf

        )

        threat = get_threat_level(

            risk

        )

        unknown = detect_unknown_threat(

            attack,

            anomaly_label[i]

        )

        category = detection_category(

            attack,

            unknown,

            anomaly_score[i]

        )

        rows.append({

            "sample_id":i,

            "prediction":attack,

            "confidence":round(

                conf,

                4

            ),

            "risk_score":risk,

            "threat_level":threat,

            "anomaly_score":round(

                float(anomaly_score[i]),

                4

            ),

            "unknown_threat":unknown,

            "detection_category":category

        })

    result = pd.DataFrame(rows)

    return result


# =========================================================
# MAIN PREDICT FUNCTION
# =========================================================

def predict_dataframe(df):

    X = preprocess(df)

    result = build_prediction_table(

        X,

        df

    )

    return result

# ============================================================
# MITRE LOOKUP
# ============================================================

from mitre_db import MITRE_ATTACK_DB

def get_attack_details(attack):

    info = MITRE_ATTACK_DB.get(

        attack,

        {

            "technique":"Unknown",

            "technique_id":"N/A",

            "tactic":"Unknown",

            "recommendations":[

                "Manual Investigation Required"

            ]

        }

    )

    return info


# ============================================================
# ENRICH RESULTS
# ============================================================

def enrich_results(result_df):

    tactics=[]

    techniques=[]

    technique_ids=[]

    recommendations=[]

    priorities=[]

    immediate=[]

    longterm=[]

    for _,row in result_df.iterrows():

        attack=row["prediction"]

        info=get_attack_details(attack)

        tactics.append(

            info["tactic"]

        )

        techniques.append(

            info["technique"]

        )

        technique_ids.append(

            info["technique_id"]

        )

        recommendations.append(

            " | ".join(

                info["recommendations"]

            )

        )

        if row["unknown_threat"]:

            priorities.append("P1")

            immediate.append(

                "Immediately isolate the affected host."

            )

            longterm.append(

                "Perform forensic investigation."

            )

        elif row["threat_level"]=="Critical":

            priorities.append("P1")

            immediate.append(

                "Contain attack immediately."

            )

            longterm.append(

                "Strengthen monitoring."

            )

        elif row["threat_level"]=="High":

            priorities.append("P2")

            immediate.append(

                "Investigate affected systems."

            )

            longterm.append(

                "Apply security patches."

            )

        elif row["threat_level"]=="Medium":

            priorities.append("P3")

            immediate.append(

                "Monitor activity."

            )

            longterm.append(

                "Review firewall policies."

            )

        else:

            priorities.append("P4")

            immediate.append(

                "No immediate action."

            )

            longterm.append(

                "Continue routine monitoring."

            )

    result_df["mitre_tactic"]=tactics

    result_df["mitre_technique"]=techniques

    result_df["mitre_id"]=technique_ids

    result_df["recommendations"]=recommendations

    result_df["priority"]=priorities

    result_df["immediate_action"]=immediate

    result_df["long_term_action"]=longterm

    return result_df


# ============================================================
# EXPORT
# ============================================================

def save_prediction_report(

        result_df,

        output_dir="prediction_results"

):

    os.makedirs(

        output_dir,

        exist_ok=True

    )

    csv_path=os.path.join(

        output_dir,

        "prediction_report.csv"

    )

    json_path=os.path.join(

        output_dir,

        "prediction_report.json"

    )

    result_df.to_csv(

        csv_path,

        index=False

    )

    result_df.to_json(

        json_path,

        orient="records",

        indent=4

    )

    return csv_path,json_path


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def predict_csv(csv_file):

    df=load_csv(csv_file)

    X=preprocess(df)

    result=build_prediction_table(

        X,

        df

    )

    result=enrich_results(

        result

    )

    save_prediction_report(

        result

    )

    return result


# ============================================================
# TEST
# ============================================================

if __name__=="__main__":

    print()

    print("="*60)

    print("ThreatVision AI Prediction Engine")

    print("="*60)

    print()

    print(

        "Run predict_csv('yourfile.csv')"

    )