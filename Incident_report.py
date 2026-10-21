
import os
import json
import pandas as pd
from datetime import datetime

# =====================================================
# PATHS
# =====================================================

REPORT_DIR = r"C:\Users\Archi\OneDrive\Desktop\IntrusionDetection\reports"

RISK_FILE = os.path.join(
    REPORT_DIR,
    "risk_scores_with_anomaly.csv"
)

RECOMMENDATION_FILE = os.path.join(
    REPORT_DIR,
    "recommendations.csv"
)

OUTPUT_CSV = os.path.join(
    REPORT_DIR,
    "incident_report.csv"
)

OUTPUT_JSON = os.path.join(
    REPORT_DIR,
    "incident_report.json"
)

# =====================================================
# LOAD FILES
# =====================================================

print("=" * 60)
print("INCIDENT REPORT GENERATOR")
print("=" * 60)

print("\nLoading files...")

risk_df = pd.read_csv(RISK_FILE)
recommend_df = pd.read_csv(RECOMMENDATION_FILE)

print(f"Risk Records          : {len(risk_df)}")
print(f"Recommendation Records: {len(recommend_df)}")

# =====================================================
# MERGE DATA
# =====================================================

report_df = pd.merge(
    risk_df,
    recommend_df[
        [
            "sample_id",
            "mitre_tactic",
            "mitre_technique",
            "mitre_id",
            "recommendations"
        ]
    ],
    on="sample_id",
    how="left"
)

# =====================================================
# ADD TIMESTAMP
# =====================================================

report_df["generated_at"] = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S"
)

# =====================================================
# REORDER COLUMNS
# =====================================================

final_columns = [

    "sample_id",

    "pred_label",
    "confidence",

    "risk_score",
    "threat_level",

    "detection_category",

    "anomaly_score",

    "mitre_tactic",
    "mitre_technique",
    "mitre_id",

    "recommendations",

    "generated_at"
]

final_columns = [
    col for col in final_columns
    if col in report_df.columns
]

report_df = report_df[final_columns]

# =====================================================
# SAVE CSV
# =====================================================

report_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print(f"\nIncident CSV saved:")
print(OUTPUT_CSV)

# =====================================================
# SAVE JSON
# =====================================================

json_output = {
    "generated_at": datetime.now().isoformat(),
    "total_incidents": int(len(report_df)),
    "incidents": report_df.to_dict(
        orient="records"
    )
}

with open(
    OUTPUT_JSON,
    "w"
) as f:

    json.dump(
        json_output,
        f,
        indent=4
    )

print(f"\nIncident JSON saved:")
print(OUTPUT_JSON)

# =====================================================
# SUMMARY
# =====================================================

print("\n" + "=" * 60)
print("INCIDENT SUMMARY")
print("=" * 60)

print(
    report_df["detection_category"]
    .value_counts()
)

# =====================================================
# TOP 10 MOST CRITICAL INCIDENTS
# =====================================================

print("\n" + "=" * 60)
print("TOP CRITICAL INCIDENTS")
print("=" * 60)

critical_df = report_df.sort_values(
    by="risk_score",
    ascending=False
).head(10)

display_cols = [
    c for c in [
        "sample_id",
        "pred_label",
        "risk_score",
        "threat_level",
        "detection_category",
        "mitre_id"
    ]
    if c in critical_df.columns
]

print(
    critical_df[display_cols]
    .to_string(index=False)
)

print("\nDONE!")
