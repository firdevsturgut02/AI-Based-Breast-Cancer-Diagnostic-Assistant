"""
evaluator.py
-------------
Final Performance Evaluation and Clinical Interpretability Module.
Fixed: TypeError in custom_object loading for Keras 3.
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
MODEL_PATH = os.path.join("results", "trainer", "models", "best_model.h5")
EVAL_OUT_DIR = os.path.join("results", "evaluator")
os.makedirs(EVAL_OUT_DIR, exist_ok=True)

# =====================================================================
# 3. LOAD MODEL WITH CUSTOM OBJECTS (HATAYI ÇÖZEN KISIM)
# =====================================================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")

print("🔍 Loading model and handling Keras 3 custom objects...")

# Keras'ın 'Mean' ve 'Amax'ı doğru çağırması için lambda tanımlıyoruz
custom_objects = {
    "Mean": lambda x, **kwargs: ops.mean(x, **kwargs),
    "Amax": lambda x, **kwargs: ops.amax(x, **kwargs)
}

try:
    # custom_object_scope kullanarak yükleme
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model successfully loaded.")
except Exception as e:
    print(f"⚠️ Initial load failed, trying alternative: {e}")
    # Alternatif doğrudan yükleme metodu
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)

# =====================================================================
# 4. HARDWARE & INFERENCE ANALYSIS
# =====================================================================
gpu_devices = tf.config.list_physical_devices('GPU')
device_name = tf.test.gpu_device_name() if gpu_devices else "CPU"

def measure_performance(model, iterations=50):
    dummy_input = np.random.rand(1, 256, 256, 3).astype(np.float32)
    for _ in range(5): _ = model.predict(dummy_input, verbose=0) # Warmup
    start_time = time.time()
    for _ in range(iterations):
        _ = model.predict(dummy_input, verbose=0)
    avg_latency = ((time.time() - start_time) / iterations) * 1000
    return avg_latency, 1000.0 / avg_latency

latency, fps = measure_performance(model)
print(f"🚀 Device: {device_name} | Latency: {latency:.2f} ms | FPS: {fps:.2f}")

# =====================================================================
# 5. DATA PREPARATION & EVALUATION (CM & ROC)
# =====================================================================
# Görüntüleri topla
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
datagen = ImageDataGenerator(rescale=1.0 / 255)
test_gen = datagen.flow_from_dataframe(
    test_df, x_col="Path", y_col="Label", target_size=(256, 256),
    class_mode="categorical", batch_size=1, shuffle=False
)
class_labels = sorted(test_gen.class_indices.keys())

# Tahminler
preds = model.predict(test_gen, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes

# Confusion Matrix

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix: Test Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(EVAL_OUT_DIR, "confusion_matrix.png"), dpi=300)
plt.close()

# ROC Analysis
plt.figure(figsize=(9, 7))
y_true_bin = label_binarize(y_true, classes=range(len(class_labels)))
for i, label in enumerate(class_labels):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], preds[:, i])
    plt.plot(fpr, tpr, label=f'{label} (AUC = {auc(fpr, tpr):.3f})', lw=2)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.title("ROC Analysis")
plt.legend()
plt.savefig(os.path.join(EVAL_OUT_DIR, "roc_analysis.png"), dpi=300)
plt.close()

# =====================================================================
# 6. GRAD-CAM (Clinical Interpretation)
# =====================================================================
def get_gradcam(model, img_array):
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

# Örnek Grad-CAM kaydı
sample_idx = 0
img_arr = tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(test_df.iloc[sample_idx]["Path"], target_size=(256, 256))) / 255.0
cam = get_gradcam(model, np.expand_dims(img_arr, axis=0))
heatmap = cv2.applyColorMap(np.uint8(255 * cv2.resize(cam, (256, 256))), cv2.COLORMAP_JET)
res = cv2.addWeighted(np.uint8(255 * img_arr), 0.6, heatmap, 0.4, 0)
cv2.imwrite(os.path.join(EVAL_OUT_DIR, "gradcam_explanation.png"), cv2.cvtColor(res, cv2.COLOR_RGB2BGR))

print(f"\n✅ Evaluation complete. Results: {EVAL_OUT_DIR}/")
