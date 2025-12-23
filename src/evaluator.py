"""
evaluator.py
-------------
Comprehensive evaluation script for breast ultrasound classification model.

Features:
- Detailed classification report (Precision, Recall, F1, AUC)
- Normalized confusion matrix
- Multi-class ROC curves
- Grad-CAM visualizations for interpretability
- Publication-quality plots (auto-saved to results/)
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import json
from itertools import cycle
from sklearn.model_selection import train_test_split

# =====================================================================
# 1. PATHS AND DIRECTORIES
# =====================================================================
DATA_DIR = "Dataset_BUSI_with_GT"
MODEL_PATH = "models/final_model_cbam.h5"  # updated model filename from trainer.py
SAVE_DIR = "results"
os.makedirs(SAVE_DIR, exist_ok=True)

# =====================================================================
# 2. DATA PREPARATION
# =====================================================================
data_paths, labels = [], []
for folder in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                data_paths.append(os.path.join(folder_path, file))
                labels.append(folder)

df = pd.DataFrame({"Path": data_paths, "Label": labels})

_, temp_df = train_test_split(df, test_size=0.2, stratify=df["Label"], random_state=123)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["Label"], random_state=123)

# =====================================================================
# 3. DATA GENERATOR
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
# 4. MODEL LOAD & PREDICT
# =====================================================================
print("\n🔍 Loading model and predicting...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
preds = model.predict(test_gen, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes

# =====================================================================
# 5. CLASSIFICATION REPORT
# =====================================================================
print("\n📄 Classification Report:")
report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
print(classification_report(y_true, y_pred, target_names=class_labels))

df_report = pd.DataFrame(report).transpose()
df_report.to_csv(os.path.join(SAVE_DIR, "classification_report.csv"))
with open(os.path.join(SAVE_DIR, "classification_report.json"), "w") as f:
    json.dump(report, f, indent=4)

# =====================================================================
# 6. CONFUSION MATRIX (Normalized)
# =====================================================================
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(6, 5))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels)
plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
cm_path = os.path.join(SAVE_DIR, "confusion_matrix.jpg")
plt.savefig(cm_path, dpi=300)
plt.close()
print(f"📈 Confusion matrix saved to {cm_path}")

# =====================================================================
# 7. ROC CURVES (Multi-class)
# =====================================================================
print("\n📉 Generating ROC curves...")

y_true_bin = label_binarize(y_true, classes=range(num_classes))
fpr, tpr, roc_auc = dict(), dict(), dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], preds[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), preds.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure(figsize=(7, 6))
colors = cycle(["aqua", "darkorange", "cornflowerblue"])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f"{class_labels[i]} (AUC = {roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], "k--", lw=2)
plt.plot(fpr["micro"], tpr["micro"], color="deeppink", linestyle=":", lw=3,
         label=f"Micro-average (AUC = {roc_auc['micro']:.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curves")
plt.legend(loc="lower right")
roc_path = os.path.join(SAVE_DIR, "roc_curves.jpg")
plt.savefig(roc_path, dpi=300)
plt.close()
print(f"✅ ROC curves saved to {roc_path}")

auc_macro = np.mean(list(roc_auc.values())[:-1])
auc_micro = roc_auc["micro"]
print(f"🎯 Final Test ROC-AUC (macro): {auc_macro:.4f}")
print(f"🎯 Final Test ROC-AUC (micro): {auc_micro:.4f}")

# =====================================================================
# 8. GRAD-CAM EXPLAINABILITY (RANDOM SAMPLE)
# =====================================================================
def generate_gradcam(model, img_array, layer_name='conv5_block16_concat'):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap[0], 0) / np.max(heatmap[0])
    return heatmap

def overlay_gradcam(img_path, heatmap, intensity=0.5, cmap=cv2.COLORMAP_JET):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    heatmap = cv2.resize(heatmap, (256, 256))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cmap)
    superimposed = cv2.addWeighted(heatmap, intensity, img, 1 - intensity, 0)
    return superimposed

example_path = test_df.sample(1, random_state=123)["Path"].values[0]
img = tf.keras.preprocessing.image.load_img(example_path, target_size=(256, 256))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array / 255.0, axis=0)

print("\n🧠 Generating Grad-CAM visualization for example image...")
heatmap = generate_gradcam(model, img_array)
overlay = overlay_gradcam(example_path, heatmap)

plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Grad-CAM Visualization Example")
gradcam_path = os.path.join(SAVE_DIR, "gradcam_example.jpg")
plt.savefig(gradcam_path, dpi=300)
plt.close()
print(f"🔥 Grad-CAM saved to {gradcam_path}")

# =====================================================================
# 9. SUMMARY EXPORT
# =====================================================================
summary_dict = {
    "Macro_AUC": float(auc_macro),
    "Micro_AUC": float(auc_micro),
    "Per_Class_AUC": {class_labels[i]: float(roc_auc[i]) for i in range(num_classes)},
    "Confusion_Matrix": cm.tolist(),
}
with open(os.path.join(SAVE_DIR, "evaluation_summary.json"), "w") as f:
    json.dump(summary_dict, f, indent=4)

print("\n📊 Evaluation summary saved to evaluation_summary.json")
print("\n✅ Evaluation completed successfully.")
