"""
evaluator.py
-------------
Final Performance Evaluation and Clinical Interpretability Module.
✅ Hardware Analysis (GPU/CPU)
✅ Inference Latency & FPS Calculation
✅ Multi-class Confusion Matrix & ROC Curves
✅ Clinical Explainability with Grad-CAM
"""

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

# Import the architecture from our factory
from model_factory import get_model

# =====================================================================
# 1. PATHS & CONFIGURATION
# =====================================================================
DATA_DIR = "Dataset_BUSI_with_GT"
MODEL_WEIGHTS_PATH = os.path.join("results", "trainer", "models", "best_model.h5")
EVAL_OUT_DIR = os.path.join("results", "evaluator")
os.makedirs(EVAL_OUT_DIR, exist_ok=True)

# =====================================================================
# 2. HARDWARE & MODEL LOADING
# =====================================================================
print("\n🖥️ Analyzing Hardware...")
gpu_devices = tf.config.list_physical_devices('GPU')
device_name = "Tesla T4" if gpu_devices else "CPU"
print(f"📍 Execution Device: {device_name}")

print("\n🏗️ Building Architecture and Loading Best Weights...")
model = get_model(num_classes=3) 

if not os.path.exists(MODEL_WEIGHTS_PATH):
    raise FileNotFoundError(f"❌ Weights not found at: {MODEL_WEIGHTS_PATH}")

model.load_weights(MODEL_WEIGHTS_PATH)
print(f"✅ Successfully loaded weights from: {MODEL_WEIGHTS_PATH}")

# =====================================================================
# 3. PERFORMANCE ANALYSIS
# =====================================================================
def measure_latency(model, iterations=50):
    dummy_input = np.random.rand(1, 256, 256, 3).astype(np.float32)
    for _ in range(5): _ = model.predict(dummy_input, verbose=0)
    start_time = time.time()
    for _ in range(iterations):
        _ = model.predict(dummy_input, verbose=0)
    avg_latency = ((time.time() - start_time) / iterations) * 1000
    return avg_latency, 1000.0 / avg_latency

latency, fps = measure_latency(model)
print(f"🚀 Performance: {latency:.2f} ms per image | Throughput: {fps:.2f} FPS")

# =====================================================================
# 4. DATA PREPARATION (TEST SET)
# =====================================================================
data_paths, labels = [], []
for folder in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, folder)
    if os.path.isdir(folder_path):
        for f in os.listdir(folder_path):
            if f.lower().endswith((".png", ".jpg")) and "mask" not in f.lower():
                data_paths.append(os.path.join(folder_path, f))
                labels.append(folder)

df = pd.DataFrame({"Path": data_paths, "Label": labels})
_, temp_df = train_test_split(df, test_size=0.2, stratify=df["Label"], random_state=123)
_, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["Label"], random_state=123)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
test_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    test_df, x_col="Path", y_col="Label", target_size=(256, 256),
    class_mode="categorical", batch_size=1, shuffle=False
)
class_names = sorted(test_gen.class_indices.keys())

# =====================================================================
# 5. SCIENTIFIC VISUALIZATION
# =====================================================================
print("🧪 Generating Predictions and Metrics...")
preds = model.predict(test_gen, verbose=1)
y_pred = np.argmax(preds, axis=1)
# FIX: Explicitly convert y_true to a NumPy array to support .astype()
y_true = np.array(test_gen.classes)

# --- Confusion Matrix ---

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix: Test Set Performance", fontsize=12, fontweight='bold')
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.savefig(os.path.join(EVAL_OUT_DIR, "confusion_matrix.png"), dpi=300)
plt.close()

# --- ROC Curves ---

plt.figure(figsize=(9, 7))
for i, label in enumerate(class_names):
    # FIX: Comparison now happens on a NumPy array
    binary_true = (y_true == i).astype(int)
    fpr, tpr, _ = roc_curve(binary_true, preds[:, i])
    plt.plot(fpr, tpr, label=f'{label} (AUC = {auc(fpr, tpr):.3f})', lw=2)

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.title("Multi-Class ROC Analysis", fontsize=12, fontweight='bold')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.2)
plt.savefig(os.path.join(EVAL_OUT_DIR, "roc_analysis.png"), dpi=300)
plt.close()

# --- Grad-CAM Explainability ---
def generate_gradcam(model, img_array):
    last_conv_layer = next(l for l in reversed(model.layers) if isinstance(l, tf.keras.layers.Conv2D))
    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]
    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.dot(conv_outputs[0], weights)
    cam = np.maximum(cam, 0)
    return cam / (np.max(cam) + 1e-10)

# Save one Grad-CAM sample
sample_path = test_df.iloc[0]["Path"]
img_raw = cv2.imread(sample_path)
img_resized = cv2.resize(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB), (256, 256))
cam = generate_gradcam(model, np.expand_dims(img_resized/255.0, axis=0))
heatmap = cv2.applyColorMap(np.uint8(255 * cv2.resize(cam, (256, 256))), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(np.uint8(255 * (img_resized/255.0)), 0.6, heatmap, 0.4, 0)
cv2.imwrite(os.path.join(EVAL_OUT_DIR, "gradcam_explanation.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

# =====================================================================
# 6. EXPORT SUMMARY
# =====================================================================
report = {
    "Device": device_name,
    "Latency_ms": round(latency, 2),
    "Throughput_FPS": round(fps, 2),
    "Test_Accuracy": float(np.mean(y_true == y_pred))
}

with open(os.path.join(EVAL_OUT_DIR, "performance_report.json"), "w") as f:
    json.dump(report, f, indent=4)

print(f"\n✅ Evaluation Complete.")
print(f"📊 Accuracy: {report['Test_Accuracy']*100:.2f}%")
print(f"📁 Files saved in: {EVAL_OUT_DIR}/")
