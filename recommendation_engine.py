import os
import json
import pandas as pd
from datetime import datetime
from mitre_db import MITRE_ATTACK_DB

# =====================================================
# PATHS
# =====================================================

REPORT_DIR = r"C:\Users\Archi\OneDrive\Desktop\IntrusionDetection\reports"

INPUT_FILE = os.path.join(REPORT_DIR, "risk_scores.csv")
OUTPUT_CSV = os.path.join(REPORT_DIR, "recommendations.csv")
OUTPUT_JSON = os.path.join(REPORT_DIR, "recommendations.json")

# =====================================================
# LOAD RISK DATA
# =====================================================

print("=" * 60)
print("MITRE ATT&CK Recommendation Engine")
print("=" * 60)

print("\nLoading risk score data...")

df = pd.read_csv(INPUT_FILE)

print(f"Total Records Loaded : {len(df)}")

# =====================================================
# GENERATE RECOMMENDATIONS
# =====================================================

records = []

for _, row in df.iterrows():

    attack = str(row["pred_label"]).strip()

    # Safe lookup
    info = MITRE_ATTACK_DB.get(attack, {})

    tactic = info.get("tactic", "Unknown")
    technique = info.get("technique", "Unknown")
    technique_id = info.get("technique_id", "N/A")

    # Accept both "recommendations" and "recommendation"
    recs = info.get(
        "recommendations",
        info.get(
            "recommendation",
            ["Investigate manually."]
        )
    )

    if isinstance(recs, list):
        recommendation_text = " | ".join(recs)
    else:
        recommendation_text = str(recs)

    records.append({

        "sample_id": row["sample_id"],

        "pred_label": attack,

        "risk_score": row["risk_score"],

        "threat_level": row["threat_level"],

        "mitre_tactic": tactic,

        "mitre_technique": technique,

        "mitre_id": technique_id,

        "recommendation": recommendation_text

    })

# =====================================================
# CREATE DATAFRAME
# =====================================================

recommendation_df = pd.DataFrame(records)

# =====================================================
# SAVE CSV
# =====================================================

recommendation_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\nRecommendations CSV Saved:")
print(OUTPUT_CSV)

# =====================================================
# SAVE JSON
# =====================================================

json_output = {

    "generated_at": datetime.now().isoformat(),

    "total_records": len(recommendation_df),

    "recommendations": records

}

with open(OUTPUT_JSON, "w") as f:

    json.dump(
        json_output,
        f,
        indent=4
    )

print("\nRecommendations JSON Saved:")
print(OUTPUT_JSON)

# =====================================================
# DISPLAY SAMPLE
# =====================================================

print("\n" + "=" * 60)
print("SAMPLE RECOMMENDATIONS")
print("=" * 60)

for item in records[:5]:

    print(f"""
Sample ID       : {item['sample_id']}
Predicted Label : {item['pred_label']}
Risk Score      : {item['risk_score']}
Threat Level    : {item['threat_level']}

MITRE Tactic    : {item['mitre_tactic']}
MITRE Technique : {item['mitre_technique']}
MITRE ID        : {item['mitre_id']}

Recommendation:
{item['recommendation']}

------------------------------------------------------------
""")

print("\nRecommendation generation completed successfully!")