import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import os
from predict import predict_csv
import tempfile
import time
from PIL import Image

from streamlit_option_menu import option_menu

from theme import load_css

from utils import *

SHAP_DIR = r"C:\Users\Archi\OneDrive\Desktop\IntrusionDetection\models\shap_plots"

REPORT_DIR = r"C:\Users\Archi\OneDrive\Desktop\IntrusionDetection\reports"


st.set_page_config(

    page_title="ThreatVision AI",

    page_icon="🛡",

    layout="wide"

)

st.markdown(

    load_css(),

    unsafe_allow_html=True

)

# ----------------------------------------------------

risk_df = load_risk()

incident_df = load_incidents()

recommend_df = load_recommendations()

summary = load_summary()

# ----------------------------------------------------

with st.sidebar:

    selected = option_menu(

        "ThreatVision AI",

        [

            "Dashboard",

            "Threat Analytics",

            "Explainability",

            "MITRE ATT&CK",

            "Incident Reports",

            "Live Prediction",

            "About"

        ],

        icons=[

            "house",

            "bar-chart",

            "robot",

            "shield",

            "file-earmark",

            "upload",

            "info-circle"

        ],

        default_index=0

    )

# ----------------------------------------------------

st.markdown(

'<p class="main-title">🛡 ThreatVision AI</p>',

unsafe_allow_html=True

)

st.markdown(

'<p class="sub-title">AI Powered Network Intrusion Detection & Threat Intelligence Platform</p>',

unsafe_allow_html=True

)

st.divider()

# ----------------------------------------------------

if selected=="Dashboard":

    st.markdown(

    '<p class="section-title">System Overview</p>',

    unsafe_allow_html=True

    )

    col1,col2,col3,col4=st.columns(4)

    with col1:

        st.metric(

            "Total Samples",

            len(risk_df)

        )

    with col2:

        st.metric(

            "Threats",

            len(risk_df[risk_df["pred_label"]!="BENIGN"])

        )

    with col3:

        st.metric(

            "Unknown Threats",

            risk_df["unknown_threat"].sum()

        )

    with col4:

        st.metric(

            "Average Risk",

            round(risk_df["risk_score"].mean(),2)

        )

    st.success("ThreatVision AI is Active")

elif selected=="Threat Analytics":

    st.markdown(
        '<p class="section-title">Threat Analytics Dashboard</p>',
        unsafe_allow_html=True
    )

    st.write("Interactive analytics of detected cyber threats.")

    st.divider()

    # ------------------------------------
    # Row 1
    # ------------------------------------

    col1,col2=st.columns(2)

    with col1:

        attack_counts = (
            risk_df["pred_label"]
            .value_counts()
            .reset_index()
        )

        attack_counts.columns = [
            "Attack",
            "Count"
        ]

        fig = px.pie(

            attack_counts,

            names="Attack",

            values="Count",

            title="Attack Distribution",

            hole=0.45

        )

        fig.update_layout(
            template="plotly_dark",
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        level = (
            risk_df["threat_level"]
            .value_counts()
            .reset_index()
        )

        level.columns=[
            "Threat",
            "Count"
        ]

        fig = px.bar(

            level,

            x="Threat",

            y="Count",

            color="Threat",

            title="Threat Level Distribution"

        )

        fig.update_layout(
            template="plotly_dark",
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # ------------------------------------
    # Row 2
    # ------------------------------------

    col3,col4=st.columns(2)

    with col3:

        detect = (

            risk_df["detection_category"]

            .value_counts()

            .reset_index()

        )

        detect.columns=[
            "Category",
            "Count"
        ]

        fig=px.bar(

            detect,

            x="Category",

            y="Count",

            color="Category",

            title="Detection Categories"

        )

        fig.update_layout(

            template="plotly_dark",

            height=450

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    with col4:

        fig=px.histogram(

            risk_df,

            x="risk_score",

            nbins=25,

            title="Risk Score Distribution"

        )

        fig.update_layout(

            template="plotly_dark",

            height=450

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    st.divider()

    # ------------------------------------
    # Row 3
    # ------------------------------------

    st.subheader("Risk Score vs Model Confidence")

    fig=px.scatter(

        risk_df,

        x="confidence",

        y="risk_score",

        color="threat_level",

        hover_data=[

            "pred_label",

            "detection_category"

        ],

        title="Risk vs Confidence"

    )

    fig.update_layout(

        template="plotly_dark",

        height=500

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.divider()

    # ------------------------------------
    # Recent Threat Table
    # ------------------------------------

    st.subheader("Recent Threats")

    show_cols=[

        "sample_id",

        "pred_label",

        "confidence",

        "risk_score",

        "threat_level",

        "detection_category"

    ]

    show_cols=[
        c for c in show_cols
        if c in risk_df.columns
    ]

    st.dataframe(

        risk_df[show_cols].head(30),

        use_container_width=True

    )

elif selected=="Explainability":

    st.markdown(
        '<p class="section-title">AI Explainability (SHAP)</p>',
        unsafe_allow_html=True
    )

    st.write(
        "Understand why the AI predicted a particular attack."
    )

    shap_files = sorted(
        [
            f for f in os.listdir(SHAP_DIR)
            if f.endswith(".png")
        ]
    )

    if len(shap_files)==0:

        st.warning(
            "No SHAP plots found."
        )

    else:

        selected_plot = st.selectbox(

            "Choose Attack",

            shap_files

        )

        image = Image.open(

            os.path.join(

                SHAP_DIR,

                selected_plot

            )

        )

        st.image(

            image,

            use_container_width=True,

            caption=selected_plot

        )

        st.info("""

Higher SHAP values indicate features that strongly influenced
the prediction for this attack.

""")

elif selected=="MITRE ATT&CK":

    st.markdown(
        '<p class="section-title">MITRE ATT&CK Explorer</p>',
        unsafe_allow_html=True
    )

    st.write(
        "Mapped ATT&CK techniques and recommended mitigations."
    )

    attacks = sorted(

        recommend_df["pred_label"].unique()

    )

    attack = st.selectbox(

        "Select Attack",

        attacks

    )

    data = recommend_df[

        recommend_df["pred_label"]==attack

    ]

    st.dataframe(

        data,

        use_container_width=True

    )

    csv = data.to_csv(index=False).encode()

    st.download_button(

        "Download Attack Report",

        csv,

        file_name=f"{attack}_mitre.csv",

        mime="text/csv"

    )

elif selected=="Incident Reports":

    st.markdown(
        '<p class="section-title">Incident Reports</p>',
        unsafe_allow_html=True
    )

    search = st.text_input(

        "Search Attack"

    )

    threat = st.selectbox(

        "Threat Level",

        ["All"] +

        sorted(

            incident_df["threat_level"].unique()

        )

    )

    df = incident_df.copy()

    if search != "":

        df = df[

            df["pred_label"]

            .str.contains(

                search,

                case=False

            )

        ]

    if threat != "All":

        df = df[

            df["threat_level"]==threat

        ]

    st.dataframe(

        df,

        use_container_width=True,

        height=500

    )

    csv = df.to_csv(index=False).encode()

    st.download_button(

        "Download Incident Report",

        csv,

        "incident_report.csv",

        "text/csv"

    )

    st.subheader("Top Critical Incidents")

    top = df.sort_values(

        "risk_score",

        ascending=False

    ).head(10)

    st.table(

        top[
            [
                "sample_id",

                "pred_label",

                "risk_score",

                "threat_level"
            ]
        ]

    )

elif selected == "Live Prediction":

    st.markdown(
        '<p class="section-title">Live Network Traffic Analysis</p>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload a CIC-IDS2017 compatible CSV file for real-time intrusion detection."
    )

    uploaded_file = st.file_uploader(
        "Choose CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        st.success("CSV Uploaded Successfully")

        if st.button("🛡 Analyze Traffic", use_container_width=True):

            with st.spinner("ThreatVision AI is analyzing network traffic..."):

                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".csv"
                ) as tmp:

                    tmp.write(uploaded_file.getbuffer())

                    temp_path = tmp.name

                progress = st.progress(0)

                for i in range(100):

                    time.sleep(0.01)

                    progress.progress(i + 1)

                result = predict_csv(temp_path)

            st.success("Analysis Complete")

            st.divider()

            st.subheader("Prediction Summary")

            c1, c2, c3, c4 = st.columns(4)

            with c1:

                st.metric(
                    "Total Samples",
                    len(result)
                )

            with c2:

                st.metric(
                    "Attacks",
                    len(
                        result[
                            result["prediction"] != "Benign"
                        ]
                    )
                )

            with c3:

                st.metric(
                    "Unknown Threats",
                    int(
                        result["unknown_threat"].sum()
                    )
                )

            with c4:

                st.metric(
                    "Average Risk",
                    round(
                        result["risk_score"].mean(),
                        2
                    )
                )

            st.divider()

            st.subheader("Prediction Results")

            st.dataframe(
                result,
                use_container_width=True,
                height=500
            )

            st.divider()

            st.subheader("Threat Level Distribution")

            threat = (
                result["threat_level"]
                .value_counts()
                .reset_index()
            )

            threat.columns = [
                "Threat Level",
                "Count"
            ]

            fig = px.pie(
                threat,
                names="Threat Level",
                values="Count",
                hole=0.45,
                title="Threat Levels"
            )

            fig.update_layout(
                template="plotly_dark"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.divider()

            st.subheader("Top 20 High-Risk Incidents")

            top = result.sort_values(
                "risk_score",
                ascending=False
            ).head(20)

            st.dataframe(
                top,
                use_container_width=True
            )

            st.divider()

            csv = result.to_csv(index=False).encode()

            st.download_button(
                "📥 Download Prediction Report",
                csv,
                "prediction_report.csv",
                "text/csv",
                use_container_width=True
            )

            st.success("ThreatVision AI analysis completed successfully.")

else:

    st.header("About ThreatVision AI")

    st.write("""

ThreatVision AI is an AI-powered Network Intrusion Detection and Threat Intelligence Platform.

Features:

• XGBoost Attack Detection

• Risk Scoring

• SHAP Explainability

• MITRE ATT&CK Mapping

• Recommendation Engine

• Isolation Forest

• Incident Reports

• Real-Time Prediction

""")