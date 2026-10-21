import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
# ==========================================
# PATHS
# ==========================================
OUTPUT_DIR = r"C:\Users\Archi\OneDrive\Desktop\IntrusionDetection\processed"
MODEL_DIR = r"C:\Users\Archi\OneDrive\Desktop\IntrusionDetection\models"
# ==========================================
# LOAD DATA
# ==========================================
print("=" * 60)
print("Loading processed data...")
print("=" * 60)
X_train = np.load(os.path.join(OUTPUT_DIR, "X_train.npy"))
X_test  = np.load(os.path.join(OUTPUT_DIR, "X_test.npy"))
y_train = np.load(os.path.join(OUTPUT_DIR, "y_train.npy"))
y_test  = np.load(os.path.join(OUTPUT_DIR, "y_test.npy"))
label_map = joblib.load(
    os.path.join(MODEL_DIR, "label_map.pkl")
)
print(f"X_train Shape : {X_train.shape}")
print(f"X_test Shape  : {X_test.shape}")
# ==========================================
# TRAIN XGBOOST MODEL
# ==========================================
print("\nTraining XGBoost Model...\n")
model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(label_map),
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method="hist",
    eval_metric="mlogloss"
)
model.fit(X_train, y_train)
print("\nModel Training Complete!")
# ==========================================
# PREDICTIONS
# ==========================================
print("\nEvaluating Model...\n")
y_pred  = model.predict(X_test)
# ==========================================
# SAVE PREDICTIONS FOR RISK SCORING
# ==========================================
y_proba = model.predict_proba(X_test)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"),  y_test)
np.save(os.path.join(OUTPUT_DIR, "y_pred.npy"),  y_pred)
np.save(os.path.join(OUTPUT_DIR, "y_proba.npy"), y_proba)
print("✅ y_test.npy, y_pred.npy, y_proba.npy saved!")
# ==========================================
# ACCURACY
# ==========================================
accuracy = accuracy_score(y_test, y_pred)
print("=" * 60)
print(f"Accuracy : {accuracy * 100:.2f}%")
print("=" * 60)
# ==========================================
# CLASSIFICATION REPORT
# ==========================================
class_names = list(label_map.values())
print("\nClassification Report:\n")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=class_names
    )
)
# ==========================================
# CONFUSION MATRIX
# ==========================================
print("\nGenerating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=False,
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
cm_path = os.path.join(
    MODEL_DIR,
    "confusion_matrix.png"
)
plt.savefig(cm_path, dpi=300)
plt.show()
print(f"Confusion Matrix Saved: {cm_path}")
# ==========================================
# FEATURE IMPORTANCE
# ==========================================
print("\nGenerating Feature Importance Plot...")
feature_names = joblib.load(
    os.path.join(MODEL_DIR, "feature_names.pkl")
)
importance  = model.feature_importances_
sorted_idx  = np.argsort(importance)[::-1]
top_n = 20
plt.figure(figsize=(10, 8))
plt.barh(
    range(top_n),
    importance[sorted_idx][:top_n]
)
plt.yticks(
    range(top_n),
    np.array(feature_names)[sorted_idx][:top_n]
)
plt.gca().invert_yaxis()
plt.title("Top 20 Important Features")
plt.tight_layout()
importance_path = os.path.join(
    MODEL_DIR,
    "feature_importance.png"
)
plt.savefig(importance_path, dpi=300)
plt.show()
print(f"Feature Importance Saved: {importance_path}")
# ==========================================
# SAVE MODEL
# ==========================================
model_path = os.path.join(
    MODEL_DIR,
    "xgboost_model.pkl"
)
joblib.dump(model, model_path)
print("\nModel Saved Successfully!")
print(f"Location : {model_path}")
print("\nDONE!")