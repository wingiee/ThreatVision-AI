import shap
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd

print("SHAP Version:", shap.__version__)
print("Import successful!")

OUTPUT_DIR = r"C:\Users\Archi\OneDrive\Desktop\IntrusionDetection\processed"
MODEL_DIR  = r"C:\Users\Archi\OneDrive\Desktop\IntrusionDetection\models"
REPORT_DIR = r"C:\Users\Archi\OneDrive\Desktop\IntrusionDetection\reports"
SHAP_DIR   = os.path.join(MODEL_DIR, "shap_plots")

os.makedirs(SHAP_DIR,   exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

print("Paths set!")

model         = joblib.load(os.path.join(MODEL_DIR, "xgboost_model.pkl"))
feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
label_map     = joblib.load(os.path.join(MODEL_DIR, "label_map.pkl"))

X_test = np.load(os.path.join(OUTPUT_DIR, "X_test.npy"))
y_test = np.load(os.path.join(OUTPUT_DIR, "y_test.npy"))
y_pred = np.load(os.path.join(OUTPUT_DIR, "y_pred.npy"))

class_names = [label_map[i] for i in range(len(label_map))]

print(f"X_test shape : {X_test.shape}")
print(f"Classes      : {class_names}")


np.random.seed(42)
sample_idx = np.random.choice(len(X_test), size=1000, replace=False)
X_sample   = X_test[sample_idx]

print(f"X_sample shape : {X_sample.shape}")
print("Sampling done!")


print("Creating SHAP TreeExplainer...")

explainer   = shap.TreeExplainer(model)

print("Calculating SHAP values... (1-2 minutes wait karo)")

explanation = explainer(X_sample)
shap_values = explanation.values

print(f"SHAP values shape : {shap_values.shape}")
print("Done!")


np.save(os.path.join(MODEL_DIR, "shap_values.npy"), shap_values)
np.save(os.path.join(MODEL_DIR, "X_sample.npy"),    X_sample)
joblib.dump(explainer, os.path.join(MODEL_DIR, "shap_explainer.pkl"))

print("shap_values.npy    saved!")
print("X_sample.npy       saved!")
print("shap_explainer.pkl saved!")


for class_idx, class_name in enumerate(class_names):
    print(f"Processing : {class_name}")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values[:, :, class_idx],
        X_sample,
        feature_names = feature_names,
        max_display   = 15,
        show          = False,
        plot_type     = "dot"
    )
    plt.title(f"SHAP Summary — {class_name}")
    plt.tight_layout()

    safe_name = class_name.replace(" ", "_").replace("/", "_")
    path      = os.path.join(SHAP_DIR, f"shap_summary_{safe_name}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved : {path}")

print("\nAll plots saved!")