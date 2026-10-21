import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_DIR = r"C:\Users\Archi\Downloads\GeneratedLabelledFlows\TrafficLabelling"    # Folder where you put CIC-IDS2017 CSVs
OUTPUT_DIR = r"C:\Users\Archi\OneDrive\Desktop\IntrusionDetection\processed"
MODEL_DIR  = r"C:\Users\Archi\OneDrive\Desktop\IntrusionDetection\models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Map every raw label → clean category
LABEL_MAP = {
    "BENIGN"            : "Benign",
    "DDoS"              : "DDoS",
    "DoS Hulk"          : "DoS",
    "DoS GoldenEye"     : "DoS",
    "DoS slowloris"     : "DoS",
    "DoS Slowhttptest"  : "DoS",
    "Heartbleed"        : "DoS",
    "PortScan"          : "Port Scan",
    "FTP-Patator"       : "Brute Force",
    "SSH-Patator"       : "Brute Force",
    "Bot"               : "Bot Attack",
    "Web Attack – Brute Force" : "Web Attack",
    "Web Attack – XSS"         : "Web Attack",
    "Web Attack – Sql Injection": "Web Attack",
    "Infiltration"      : "Infiltration",
}

# ─────────────────────────────────────────────
# STEP 1 – LOAD ALL CSV FILES
# ─────────────────────────────────────────────
def load_dataset(data_dir: str) -> pd.DataFrame:
    """
    CIC-IDS2017 comes as multiple CSV files (one per day).
    This function loads and merges all of them.
    """
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"\n[ERROR] No CSV files found in '{data_dir}'.\n"
            "Download CIC-IDS2017 from: https://www.unb.ca/cic/datasets/ids-2017.html\n"
            "Extract all CSV files into the './dataset/' folder and re-run."
        )

    print(f"[INFO] Found {len(csv_files)} CSV file(s). Loading …")
    frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(f, low_memory=False, encoding="latin-1")
            print(f"  ⚠  {os.path.basename(f)} – UTF-8 failed, re-read with latin-1")
        print(f"  ✔  {os.path.basename(f)}  →  {df.shape[0]:,} rows")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True, copy=False)
    print(f"\n[INFO] Total rows after merge: {combined.shape[0]:,}")
    return combined


# ─────────────────────────────────────────────
# STEP 2 – CLEAN COLUMN NAMES
# ─────────────────────────────────────────────
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names (CIC CSVs often have leading spaces)."""
    df.columns = df.columns.str.strip()
    return df
# ─────────────────────────────────────────────
# STEP 3 – HANDLE MISSING & INFINITE VALUES
# ─────────────────────────────────────────────
def handle_missing_infinite(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[STEP 3] Handling missing / infinite values …")

    # Replace inf / -inf with NaN, then drop those rows
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    missing_before = df.isnull().sum().sum()
    df.dropna(inplace=True)
    print(f"  Removed {missing_before:,} NaN / Inf cells. Rows remaining: {df.shape[0]:,}")
    return df


# ─────────────────────────────────────────────
# STEP 4 – LABEL CLEANING
# ─────────────────────────────────────────────
def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[STEP 4] Cleaning labels …")

    # Detect the label column (CIC-IDS2017 uses ' Label')
    label_col = None
    for col in df.columns:
        if col.lower() == "label":
            label_col = col
            break
    if label_col is None:
        raise KeyError("Could not find a 'Label' column. Check your CSV files.")

    df.rename(columns={label_col: "Label"}, inplace=True)
    df["Label"] = df["Label"].str.strip()

    # Apply the mapping; unknown labels kept as-is
    df["Label"] = df["Label"].map(LABEL_MAP).fillna(df["Label"])

    print("  Label distribution:")
    print(df["Label"].value_counts().to_string()); print()
    return df


# ─────────────────────────────────────────────
# STEP 5 – REMOVE DUPLICATES & CONSTANT COLUMNS
# ─────────────────────────────────────────────
def remove_noise(df: pd.DataFrame) -> pd.DataFrame:
    print("[STEP 5] Removing duplicates and zero-variance columns …")

    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"  Duplicates removed: {before - len(df):,}")

    # Drop columns with only one unique value (useless for ML)
    nunique = df.nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        df.drop(columns=const_cols, inplace=True)
        print(f"  Constant columns dropped: {const_cols}")
    else:
        print("  No constant columns found.")
    return df


# ─────────────────────────────────────────────
# STEP 6 – FEATURE SELECTION
# ─────────────────────────────────────────────
# These are the top features used in most CIC-IDS2017 research papers.
# You can add more or use all columns — but this keeps training fast.
SELECTED_FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
]

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[STEP 6] Selecting features …")
    available = [f for f in SELECTED_FEATURES if f in df.columns]
    missing   = [f for f in SELECTED_FEATURES if f not in df.columns]

    if missing:
        print(f"  Warning: {len(missing)} features not found in data (skipped): {missing[:5]} …")

    df = df[available + ["Label"]].copy()
    print(f"  Using {len(available)} features + Label column.")
    return df


# ─────────────────────────────────────────────
# STEP 7 – ENCODE LABELS
# ─────────────────────────────────────────────
def encode_labels(df: pd.DataFrame):
    """
    Returns:
      df        – dataframe with numeric 'Label_Enc' column added
      le        – fitted LabelEncoder (save this for inference)
      label_map – dict {encoded_int: class_name}
    """
    print("\n[STEP 7] Encoding labels …")
    le = LabelEncoder()
    df["Label_Enc"] = le.fit_transform(df["Label"])

    label_map = {i: name for i, name in enumerate(le.classes_)}
    print("  Encoded classes:", label_map)

    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    print(f"  LabelEncoder saved → {MODEL_DIR}label_encoder.pkl")
    return df, le, label_map


# ─────────────────────────────────────────────
# STEP 8 – HANDLE CLASS IMBALANCE (optional)
# ─────────────────────────────────────────────
def balance_classes(df: pd.DataFrame, strategy: str = "undersample",
                    max_per_class: int = 50_000) -> pd.DataFrame:
    """
    strategy:
      'undersample' – cap each class at max_per_class  (fast, recommended)
      'none'        – keep original distribution
    """
    if strategy == "none":
        return df

    print(f"\n[STEP 8] Balancing classes (strategy={strategy}, max={max_per_class:,}) …")
    frames = []
    for label, group in df.groupby("Label"):
        if len(group) > max_per_class:
            group = resample(group, replace=False, n_samples=max_per_class, random_state=42)
        frames.append(group)

    balanced = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=42)
    print("  Balanced distribution:")
    print(balanced["Label"].value_counts().to_string()); print()
    return balanced


# ─────────────────────────────────────────────
# STEP 9 – SCALE FEATURES
# ─────────────────────────────────────────────
def scale_features(X_train: np.ndarray, X_test: np.ndarray):
    """
    Fit scaler on training data only → prevent data leakage.
    Returns scaled arrays + fitted scaler.
    """
    print("[STEP 9] Scaling features (StandardScaler) …")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    print(f"  Scaler saved → {MODEL_DIR}scaler.pkl")
    return X_train_scaled, X_test_scaled, scaler


# ─────────────────────────────────────────────
# STEP 10 – SPLIT DATA
# ─────────────────────────────────────────────
def split_data(df: pd.DataFrame, feature_cols: list, test_size: float = 0.2):
    print(f"\n[STEP 10] Splitting data (test_size={test_size}) …")
    X = df[feature_cols].values
    y = df["Label_Enc"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# VISUALISATION HELPERS
# ─────────────────────────────────────────────
def plot_class_distribution(df: pd.DataFrame, title: str = "Class Distribution"):
    plt.figure(figsize=(10, 5))
    ax = df["Label"].value_counts().plot(kind="bar", color="steelblue", edgecolor="black")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Attack Type")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "class_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved → {path}")


def plot_correlation_heatmap(df: pd.DataFrame, n_features: int = 20):
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:n_features]
    plt.figure(figsize=(14, 10))
    sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", annot=False, linewidths=0.3)
    plt.title("Feature Correlation Heatmap (top features)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved → {path}")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def run_pipeline():
    print("=" * 60)
    print("  IDS Preprocessing Pipeline – CIC-IDS2017")
    print("=" * 60)

    # 1. Load
    df = load_dataset(DATA_DIR)

    # 2. Clean column names
    df = clean_columns(df)

    # 3. Missing / Inf
    df = handle_missing_infinite(df)

    # 4. Labels
    df = clean_labels(df)

    # 5. Duplicates & constants
    df = remove_noise(df)

    # 6. Feature selection
    df = select_features(df)

    # Visualise raw distribution
    print("\n[INFO] Generating plots …")
    plot_class_distribution(df, "Raw Class Distribution (after cleaning)")
    plot_correlation_heatmap(df)

    # 7. Encode labels
    df, le, label_map = encode_labels(df)

    # 8. Balance classes
    df = balance_classes(df, strategy="undersample", max_per_class=50_000)
    plot_class_distribution(df, "Balanced Class Distribution")

    # 9 & 10. Split + scale
    feature_cols = [c for c in df.columns if c not in ("Label", "Label_Enc")]
    X_train, X_test, y_train, y_test = split_data(df, feature_cols)
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

    # Save processed splits
    print("\n[INFO] Saving processed data …")
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train_sc)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"),  X_test_sc)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"),  y_test)

    # Also save feature names for later use (SHAP, dashboard)
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_names.pkl"))
    joblib.dump(label_map,    os.path.join(MODEL_DIR, "label_map.pkl"))

    print("\n[DONE] Preprocessing complete!")
    print(f"  Processed arrays  → {OUTPUT_DIR}")
    print(f"  Scaler / Encoder  → {MODEL_DIR}")
    print("\nNext step: run  train_model.py  to train XGBoost.")


# ─────────────────────────────────────────────
# UTILITY: load processed data in other scripts
# ─────────────────────────────────────────────
def load_processed():
    """Call this from train_model.py to load the saved arrays."""
    X_train = np.load(os.path.join(OUTPUT_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(OUTPUT_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(OUTPUT_DIR, "y_train.npy"))
    y_test  = np.load(os.path.join(OUTPUT_DIR, "y_test.npy"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    label_map    = joblib.load(os.path.join(MODEL_DIR, "label_map.pkl"))
    scaler       = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    le           = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    return X_train, X_test, y_train, y_test, feature_cols, label_map, scaler, le


if __name__ == "__main__":
    run_pipeline()
