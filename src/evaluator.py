"""
evaluator.py
-------------
Final Model Evaluation and Clinical Interpretability Module
for Breast Ultrasound Image Classification (Academic Edition).

Version: 2.0
Core Features:
    ✅ Hardware Benchmarking (GPU/CPU Profiling)
    ✅ Inference Latency & FPS Measurement
    ✅ Multi-Class Confusion Matrix & ROC Curves
    ✅ Grad-CAM Visualization for Clinical Explainability
    ✅ Structured JSON Report for Publication & Archiving

Author: [Your Name]
Affiliation: [Your Institution]
Date: [Auto-generated]
"""

# =====================================================================
# 1. IMPORTS
# =====================================================================
import os
import time
import json
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from datetime import datetime

from model_factory import get_model  # Import custom architecture


# =====================================================================
# 2. CONFIGURATION
# =====================================================================
DATA_DIR = "Dataset_BUSI_with_GT"
MODEL_WEIGHTS_PATH = os.path.join("results", "trainer", "models", "best_model.h5")
EVAL_OUT_DIR = os.path.join("results", "evaluator")
os.makedirs(EVAL_OUT_DIR, exist_ok=True)

# Logging utility
def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# =====================================================================
# 3. HARDWARE INSPECTION & MODEL LOADING
# =====================================================================
log("🖥️ Hardware Analysis in Progress...")
gpu_devices = tf.config.list_physical_devices('GPU')
device_name = tf.test.gpu_device_name() if gpu_devices else "CPU"
log(f"📍 Execution Device: {device_name}")

log("🏗️ Building Model Architecture and Loading Weights...")
model = get_model(num_classes=3)

if not os.path.exists(MODEL_WEIGHTS_PATH):
    raise FileNotFoundError(f"❌ Model weights not found: {MODEL_WEIGHTS_PATH}")

model.load_weights(MODEL_WEIGHTS_PATH)
log("✅ Successfully loaded pre-trained weights.")


# =====================================================================
# 4. PERFORMANCE BENCHMARKING
# =====================================================================
def measure_latency(model: tf.keras.Model, iterations: int = 50):
    """Measure average inference latency and compute throughput (FPS)."""
    dummy_input = np.random.rand(1, 256, 256, 3).astype(np.float32)
    for _ in range(5):  # Warmup
        _ = model.predict(dummy_input, verbose=0)
    start_time = time.time()
    for _ in range(iterations):
        _ = model.predict(dummy_input, verbose=0)
    elapsed = time.time() - start_time
    latency = (elapsed / iterations) * 1000  # milliseconds
    fps = 1000.0 / latency
    return latency, fps

latency, fps = measure_latency(model)
log(f"🚀 Model Performance → {latency:.2f} ms / image  |  {fps:.2f} FPS")


# =====================================================================
# 5. TEST DATA PREPARATION
# =====================================================================
data_paths, labels = [], []
for folder in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, folder)
    if os.path.isdir(folder_path):
        for f in os.listdir(folder_path):
            if f.lower().endswith((".png", ".jpg", ".jpeg")) and "mask" not in f.lower():
                data_paths.append(os.path.join(folder_path, f))
                labels.append(folder)

df = pd.DataFrame({"Path": data_paths, "Label": labels})

# Stratified Split (same logic as training)
_, temp_df = train_test_split(df, test_size=0.2, stratify=df["Label"], random_state=123)
_, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["Label"], random_state=123)

# Image generator (standardized normalization)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
test_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_dataframe(
    test_df,
    x_col="Path",
    y_col="Label",
    target_size=(256, 256),
    class_mode="categorical",
    batch_size=1,
    shuffle=False
)

class_names = sorted(test_gen.class_indices.keys())
log(f"📊 Evaluation classes detected: {class_names}")


# =====================================================================
# 6. MODEL EVALUATION & VISUALIZATION
# =====================================================================
log("🧪 Generating predictions on test set...")
preds = model.predict(test_gen, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = np.array(test_gen.classes)

# --- Confusion Matrix ---
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix (Test Set)", fontsize=13, fontweight='bold')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(os.path.join(EVAL_OUT_DIR, "confusion_matrix.png"), dpi=300)
plt.close()
log("📉 Confusion matrix saved.")

# --- ROC Curves ---
plt.figure(figsize=(9, 7))
for i, label in enumerate(class_names):
    binary_true = (y_true == i).astype(int)
    fpr, tpr, _ = roc_curve(binary_true, preds[:, i])
    plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {auc(fpr, tpr):.3f})')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.title("Multi-Class ROC Analysis", fontsize=13, fontweight='bold')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(EVAL_OUT_DIR, "roc_analysis.png"), dpi=300)
plt.close()
log("📊 ROC analysis saved.")


# =====================================================================
# 7. GRAD-CAM VISUALIZATION
# =====================================================================
def generate_gradcam(model: tf.keras.Model, img_array: np.ndarray) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for visual interpretability.

    Args:
        model: Trained Keras model.
        img_array: Input image (1, H, W, 3) normalized to [0, 1].
    Returns:
        np.ndarray: Normalized heatmap.
    """
    # Find last convolutional layer
    last_conv = next(layer for layer in reversed(model.layers) if isinstance(layer, tf.keras.layers.Conv2D))
    grad_model = tf.keras.models.Model([model.inputs], [last_conv.output, model.output])

    with tf.GradientTape() as tape:
        conv_out, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]
    grads = tape.gradient(loss, conv_out)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.dot(conv_out[0], weights)
    cam = np.maximum(cam, 0)
    cam /= np.max(cam) + 1e-10
    return cam

# Example Grad-CAM visualization
log("🧠 Generating Grad-CAM interpretability map...")
sample_path = test_df.iloc[0]["Path"]
img_raw = cv2.imread(sample_path)
img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (256, 256))
input_array = np.expand_dims(img_resized / 255.0, axis=0)

cam = generate_gradcam(model, input_array)
heatmap = cv2.applyColorMap(np.uint8(255 * cv2.resize(cam, (256, 256))), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)
cv2.imwrite(os.path.join(EVAL_OUT_DIR, "gradcam_explanation.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
log("📈 Grad-CAM heatmap generated and saved.")


# =====================================================================
# 8. EXPORT STRUCTURED REPORT
# =====================================================================
test_accuracy = float(np.mean(y_true == y_pred))
report = {
    "Evaluation_Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Device": device_name,
    "Latency_ms": round(latency, 2),
    "Throughput_FPS": round(fps, 2),
    "Test_Accuracy": test_accuracy,
    "Num_Test_Samples": int(len(test_df)),
    "Classes": class_names
}

report_path = os.path.join(EVAL_OUT_DIR, "performance_report.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=4)

log("📜 Evaluation report exported successfully.")
log(f"✅ Test Accuracy: {test_accuracy * 100:.2f}%")
log(f"📁 All evaluation outputs saved under: {EVAL_OUT_DIR}/")
