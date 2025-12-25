"""
evaluator.py
-------------
Comprehensive evaluation script for breast ultrasound classification.
Updates:
✅ Outputs saved to results/evaluator/
✅ Model path synced with trainer.py (results/trainer/models/)
✅ Grad-CAM layer targeting improved for DenseNet121+CBAM
"""

# =====================================================================
# 1. IMPORTS
# =====================================================================
import os
import json
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from itertools import cycle

# =====================================================================
# 2. PATHS & DIRECTORIES
# =====================================================================
DATA_DIR = "Dataset_BUSI_with_GT"
# Trainer modülü tarafından kaydedilen en iyi modeli hedefler
MODEL_PATH = os.path.join("results", "trainer", "models", "best_model_cbam.h5")

# Çıktılar için özel klasör
EVAL_OUT_DIR = os.path.join("results", "evaluator")
os.makedirs(EVAL_OUT_DIR, exist_ok=True)

# =====================================================================
# 3. DATA PREPARATION
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

# Trainer ile aynı seed (123) kullanılarak test setinin ayrılması
_, temp_df = train_test_split(
    df, test_size=0.2, stratify=df["Label"], random_state=123
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["Label"], random_state=123
)

# =====================================================================
# 4. DATA GENERATOR
# =====================================================================
datagen = ImageDataGenerator(rescale=1.0 / 255)

test_gen = datagen.flow_from_dataframe(
    test_df,
    x_col="Path",
    y_col="Label",
    target_size=(256, 256),
    class_mode="categorical",
    batch_size=16,
    shuffle=False
)

class_labels = list(test_gen.class_indices.keys())
num_classes = len(class_labels)

# =====================================================================
# 5. LOAD MODEL & PREDICT
# =====================================================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model bulunamadı: {MODEL_PATH}. Lütfen önce trainer.py çalıştırın.")

print("\n🔍 Model yükleniyor ve tahminler oluşturuluyor...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

preds = model.predict(test_gen, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes

# =====================================================================
# 6. CLASSIFICATION REPORT
# =====================================================================
report = classification_report(
    y_true, y_pred, target_names=class_labels, output_dict=True
)

pd.DataFrame(report).transpose().to_csv(
    os.path.join(EVAL_OUT_DIR, "classification_report.csv")
)

with open(os.path.join(EVAL_OUT_DIR, "classification_report.json"), "w") as f:
    json.dump(report, f, indent=4)

# =====================================================================
# 7. CONFUSION MATRIX (NORMALIZED)
# =====================================================================
cm = confusion_matrix(y_true, y_pred)
cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=class_labels,
    yticklabels=class_labels
)
plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

plt.savefig(os.path.join(EVAL_OUT_DIR, "confusion_matrix_normalized.png"), dpi=300)
plt.close()

# =====================================================================
# 8. ROC CURVES (MULTI-CLASS)
# =====================================================================
y_true_bin = label_binarize(y_true, classes=range(num_classes))

fpr, tpr, roc_auc = {}, {}, {}
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], preds[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), preds.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure(figsize=(8, 7))
colors = cycle(["darkorange", "cornflowerblue", "green"])

for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], lw=2, label=f"{class_labels[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], "k--", lw=2)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-class ROC Curves")
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(os.path.join(EVAL_OUT_DIR, "roc_curves.png"), dpi=300)
plt.close()

# =====================================================================
# 9. GRAD-CAM (Visual Interpretability)
# =====================================================================
def generate_gradcam(model, img_array, layer_name):
    # CBAM bloğundan önceki son konvolüsyon katmanını hedeflemek en iyi sonucu verir
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(conv_out * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap[0], 0)
    heatmap /= np.max(heatmap) + 1e-8
    return heatmap

def overlay_gradcam(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    heatmap = cv2.resize(heatmap, (256, 256))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)

# Örnek bir görüntü üzerinde Grad-CAM uygula
try:
    sample_row = test_df.sample(1, random_state=42)
    sample_path = sample_row["Path"].values[0]
    img = tf.keras.preprocessing.image.load_img(sample_path, target_size=(256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # DenseNet mimarisinde uygun bir katman seçimi
    target_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            target_layer = layer.name
            break

    heatmap = generate_gradcam(model, img_array, layer_name=target_layer)
    gradcam_img = overlay_gradcam(sample_path, heatmap)

    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(gradcam_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Grad-CAM: {sample_row['Label'].values[0]}")
    plt.savefig(os.path.join(EVAL_OUT_DIR, "gradcam_sample.png"), dpi=300)
    plt.close()
    print(f"🖼️ Grad-CAM örneği kaydedildi: {EVAL_OUT_DIR}/gradcam_sample.png")
except Exception as e:
    print(f"⚠️ Grad-CAM oluşturulurken hata: {e}")

# =====================================================================
# 10. SUMMARY EXPORT
# =====================================================================
summary = {
    "Macro_AUC": float(np.mean(list(roc_auc.values()))),
    "Micro_AUC": float(roc_auc["micro"]),
    "Per_Class_AUC": {class_labels[i]: float(roc_auc[i]) for i in range(num_classes)},
    "Confusion_Matrix": cm.tolist()
}

with open(os.path.join(EVAL_OUT_DIR, "evaluation_summary.json"), "w") as f:
    json.dump(summary, f, indent=4)

print("\n✅ Değerlendirme başarıyla tamamlandı.")
print(f"📁 Tüm sonuçlar (CSV, JSON, PNG) kaydedildi: {EVAL_OUT_DIR}/")
