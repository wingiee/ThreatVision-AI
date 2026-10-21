import os
import pandas as pd
import json

REPORT_DIR = r"C:\Users\Archi\OneDrive\Desktop\IntrusionDetection\reports"

def load_incidents():

    return pd.read_csv(
        os.path.join(
            REPORT_DIR,
            "incident_report.csv"
        )
    )

def load_risk():

    return pd.read_csv(
        os.path.join(
            REPORT_DIR,
            "risk_scores_with_anomaly.csv"
        )
    )

def load_recommendations():

    return pd.read_csv(
        os.path.join(
            REPORT_DIR,
            "recommendations.csv"
        )
    )

def load_summary():

    with open(
        os.path.join(
            REPORT_DIR,
            "risk_summary.json"
        )
    ) as f:

        return json.load(f)