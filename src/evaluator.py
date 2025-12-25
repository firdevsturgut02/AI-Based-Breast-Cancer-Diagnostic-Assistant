"""
evaluator.py
-------------
Final Performance Evaluation and Clinical Interpretability Module.
Technical Features:
✅ Hardware Utilization Report (GPU/CPU)
✅ Inference Latency & Throughput (FPS) Analysis
✅ Keras 3 Custom Object Handling ('Mean', 'Amax')
✅ Publication-Ready Visuals: Confusion Matrix, ROC, Grad-CAM
✅ Model Source: results/trainer/models/best_model.h5
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
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from keras import ops

# =====================================================================
# 2. PATHS & CONFIGURATION
# =====================================================================
DATA_DIR = "Dataset_BUSI_with_GT"
# Trainer modülü tarafından kaydedilen en iyi model
MODEL_PATH = os.path.join("results", "trainer", "models", "best_model.h5")

# Çıktıların kaydedileceği özel klasör
EVAL_OUT_DIR = os.path.join("results", "evaluator")
os.makedirs(EVAL_OUT_DIR, exist_ok=True)

# =====================================================================
# 3. HARDWARE & DEVICE ANALYSIS
# =====================================================================
print("\n🖥️ Hardware Analysis...")
gpu_devices = tf.config.list_physical_devices('GPU')
device_name = tf.test.gpu_device_name() if gpu_devices else "CPU"
print(f"📍 Execution Device: {device_name}")

# =====================================================================
# 4. LOAD MODEL WITH CUSTOM OBJECTS
# =====================================================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}. Please run trainer.py first.")

print("🔍 Loading model and handling Keras 3 custom objects...")
# Keras 3 'ops' fonksiyonlarını model yükleme sırasında tanıtıyoruz
custom_objects = {"Mean": ops.mean, "Amax": ops.amax}

with tf.keras.utils.custom_object_scope(custom_objects):
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print(f"✅ Model successfully loaded from: {MODEL_PATH}")

# =====================================================================
# 5. INFERENCE TIME ANALYSIS
# =====================================================================
def measure_performance(model, iterations=100):
    print(f"⏱️ Measuring inference time over {iterations} iterations...")
    dummy_input = np.random.rand(1, 256, 256, 3).astype(np.float32)
    
    # Warm-up (İlk çalıştırma yükü)
    for _ in range(10): _ = model.predict(dummy_input, verbose=0)
    
    start_time = time.time()
    for _ in range(iterations):
        _ = model.predict(dummy_input, verbose=0)
    end_time = time.time()
    
    avg_latency = ((end_time - start_time) / iterations) * 1000 # ms
    fps = 1.0 / (avg_latency / 1000)
    return avg_latency, fps

latency, fps = measure_performance(model)
print(f"🚀 Average Latency: {latency:.2f} ms | Throughput: {fps:.2f} FPS")

# =====================================================================
# 6. DATA PREPARATION (TEST SET)
# =====================================================================
data_paths, labels = [], []
for folder in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.lower().endswith((".png", ".jpg", ".jpeg")) and "mask" not in file.lower():
                data_paths.append(os.path.join(folder_path, file))
                labels.append(folder)

df = pd.DataFrame({"Path": data_paths, "Label": labels})
_, temp_df = train_test_split(df, test_size=0.2, stratify=df["Label"], random_state=123)
_, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["Label"], random_state=123)

datagen = ImageDataGenerator(rescale=1.0 / 255)
test_gen = datagen.flow_from_dataframe(
    test_df, x_col="Path", y_col="Label", target_size=(256, 256),
    class_mode="categorical", batch_size=1, shuffle=False
)
class_labels = sorted(test_gen.class_indices.keys())

# =====================================================================
# 7. PERFORMANCE METRICS (CM & ROC)
# =====================================================================
print("📊 Generating evaluation metrics...")
preds = model.predict(test_gen, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix: Test Set", fontsize=14, fontweight='bold')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(EVAL_OUT_DIR, "confusion_matrix.png"), dpi=300)
plt.close()

# ROC Curves
plt.figure(figsize=(9, 7))
y_true_bin = label_binarize(y_true, classes=range(len(class_labels)))
for i, label in enumerate(class_labels):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], preds[:, i])
    plt.plot(fpr, tpr, label=f'{label} (AUC = {auc(fpr, tpr):.3f})', lw=2)

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.title("Multi-Class ROC Analysis", fontsize=14, fontweight='bold')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.savefig(os.path.join(EVAL_OUT_DIR, "roc_analysis.png"), dpi=300)
plt.close()

# =====================================================================
# 8. GRAD-CAM (Explainable AI)
# =====================================================================
[Image of Grad-CAM visualization on medical image]
def get_gradcam(model, img_array):
    # Son konvolüsyon katmanını otomatik bul
    last_conv_layer = next(l for l in reversed(model.layers) if isinstance(l, tf.keras.layers.Conv2D))
    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)[0]
    output = conv_outputs[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1)
    
    cam = np.maximum(cam, 0)
    cam = cam / (np.max(cam) + 1e-10)
    return cam

# Rastgele bir test örneği üzerinde görselleştirme
sample_idx = np.random.randint(len(test_df))
sample_path = test_df.iloc[sample_idx]["Path"]
img = tf.keras.preprocessing.image.load_img(sample_path, target_size=(256, 256))
img_arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
heatmap = get_gradcam(model, np.expand_dims(img_arr, axis=0))

heatmap_res = cv2.resize(heatmap, (256, 256))
heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_res), cv2.COLORMAP_JET)
superimposed = cv2.addWeighted(np.uint8(255 * img_arr), 0.6, heatmap_color, 0.4, 0)

cv2.imwrite(os.path.join(EVAL_OUT_DIR, "gradcam_explanation.png"), cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))

# =====================================================================
# 9. TECHNICAL REPORT EXPORT
# =====================================================================
tech_summary = {
    "Model_Path": MODEL_PATH,
    "Hardware_Device": device_name,
    "Inference_Latency_ms": round(latency, 2),
    "Throughput_FPS": round(fps, 2),
    "Test_Samples": len(test_df),
    "TF_Version": tf.__version__
}

with open(os.path.join(EVAL_OUT_DIR, "technical_performance.json"), "w") as f:
    json.dump(tech_summary, f, indent=4)

print(f"\n✅ Evaluation complete.")
print(f"📊 Latency: {latency:.2f} ms | FPS: {fps:.2f}")
print(f"📁 Reports saved in: {EVAL_OUT_DIR}/")
