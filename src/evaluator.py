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
# PATHS
# =====================================================================
DATA_DIR = "Dataset_BUSI_with_GT"
MODEL_PATH = os.path.join("results", "trainer", "models", "best_model.h5")
EVAL_OUT_DIR = os.path.join("results", "evaluator")
os.makedirs(EVAL_OUT_DIR, exist_ok=True)

# =====================================================================
# MODEL LOADING (FIXED FOR KERAS 3 CUSTOM OBJECTS)
# =====================================================================
print("\n🔍 Loading model and handling Keras 3 custom objects...")

# Hatayı çözmek için operasyonları açık fonksiyonlar olarak tanımlıyoruz
def keras_mean(x, **kwargs):
    return ops.mean(x, **kwargs)

def keras_amax(x, **kwargs):
    return ops.amax(x, **kwargs)

custom_objects = {
    "Mean": keras_mean,
    "Amax": keras_amax
}

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model dosyası bulunamadı: {MODEL_PATH}")

# compile=False kullanmak yükleme sırasındaki katman hatalarını önler
with tf.keras.utils.custom_object_scope(custom_objects):
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print("✅ Model başarıyla yüklendi.")

# =====================================================================
# HARDWARE & PERFORMANCE ANALYSIS
# =====================================================================
gpu_devices = tf.config.list_physical_devices('GPU')
device_name = "/device:GPU:0" if gpu_devices else "CPU"

# Inference hızı ölçümü
dummy_input = np.random.rand(1, 256, 256, 3).astype(np.float32)
for _ in range(10): _ = model.predict(dummy_input, verbose=0) # Warm-up
start = time.time()
for _ in range(100): _ = model.predict(dummy_input, verbose=0)
latency = ((time.time() - start) / 100) * 1000
fps = 1000 / latency

# =====================================================================
# TEST DATA PREPARATION
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
# RESULTS & PLOTS
# =====================================================================
print("🧪 Test seti üzerinde değerlendirme yapılıyor...")
preds = model.predict(test_gen, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes

# 1. Confusion Matrix

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues", 
            xticklabels=class_names, yticklabels=class_names)
plt.title("Karmaşıklık Matrisi (Confusion Matrix)")
plt.savefig(os.path.join(EVAL_OUT_DIR, "confusion_matrix.png"), dpi=300)
plt.close()

# 2. ROC Analysis

plt.figure(figsize=(9, 7))
y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
for i, label in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], preds[:, i])
    plt.plot(fpr, tpr, label=f'{label} (AUC = {auc(fpr, tpr):.3f})', lw=2)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.legend()
plt.title("ROC Eğrileri")
plt.savefig(os.path.join(EVAL_OUT_DIR, "roc_analysis.png"), dpi=300)
plt.close()

# 3. Grad-CAM (Klinik Açıklanabilirlik)

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

sample_idx = 0
img_arr = tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(test_df.iloc[sample_idx]["Path"], target_size=(256, 256))) / 255.0
cam = get_gradcam(model, np.expand_dims(img_arr, axis=0))
heatmap = cv2.applyColorMap(np.uint8(255 * cv2.resize(cam, (256, 256))), cv2.COLORMAP_JET)
res = cv2.addWeighted(np.uint8(255 * img_arr), 0.6, heatmap, 0.4, 0)
cv2.imwrite(os.path.join(EVAL_OUT_DIR, "gradcam_explanation.png"), cv2.cvtColor(res, cv2.COLOR_RGB2BGR))

# 4. Teknik Performans Özeti
tech_report = {
    "Best_Model_Path": MODEL_PATH,
    "Device": device_name,
    "Latency_ms": round(latency, 2),
    "Throughput_FPS": round(fps, 2),
    "Test_Accuracy": float(np.mean(y_true == y_pred))
}
with open(os.path.join(EVAL_OUT_DIR, "performance_summary.json"), "w") as f:
    json.dump(tech_report, f, indent=4)

print(f"\n✅ Evaluator tamamlandı. Çıktılar: {EVAL_OUT_DIR}/")
