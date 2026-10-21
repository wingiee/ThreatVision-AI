
# ThreatVision AI

AI-powered network intrusion detection system using
machine learning and anomaly detection.

## Features
- Network traffic classification using XGBoost
- Anomaly detection using Isolation Forest
- SHAP-based model explainability
- MITRE ATT&CK-based mitigation recommendations
- Risk scoring and incident report generation
- Streamlit dashboard for threat visualization

## Tech Stack
- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- SHAP
- Streamlit

## Project Structure
- models/ - Trained models
- reports/ - Generated reports
- recommendation_engine.py - Security recommendations
- isolation_forest.py - Anomaly detection
- mitre_db.py - MITRE ATT&CK mappings
- requirements.txt - Dependencies

## Setup

1. Clone this repository.
2. Create and activate a virtual environment.
3. Install dependencies using requirements.txt.
4. Configure the required dataset and model paths.
5. Run the relevant pipeline scripts.
6. Start the dashboard using Streamlit.

## Run Dashboard

streamlit run app.py

Replace app.py with the actual name of your dashboard file.

## Disclaimer
For educational and research purposes.
