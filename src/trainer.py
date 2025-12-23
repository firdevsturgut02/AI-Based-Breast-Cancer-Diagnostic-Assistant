"""
trainer.py
-----------
Research-grade breast ultrasound classification trainer for TÜBİTAK project.

Includes:
- Transfer learning with DenseNet121 + CBAM attention
- Adaptive class weighting / Focal Loss
- Mixed precision training
- Grad-CAM explainability
- TensorBoard logging
- Reproducible results
"""

# =====================================================================
# 1. IMPORTS
# =====================================================================
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import (
    Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Conv2D, Multiply, Add, Activation
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.metrics import confusion_matrix, roc_curve, auc

# =====================================================================
# 2. REPRODUCIBILITY
# =====================================================================
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Mixed precision for performance (uses float16 on GPU)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# =====================================================================
# 3. PATHS
# =====================================================================
DATA_DIR = "Dataset_BUSI_with_GT"
OUTPUT_DIR = "models"
RESULTS_DIR = "results"
LOG_DIR = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# =====================================================================
# 4. DATA PREPARATION
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

train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["Label"], random_state=SEED)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["Label"], random_state=SEED)

print(f"✅ Data prepared: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

# Compute class weights to handle imbalance
class_names = sorted(df["Label"].unique())

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array(class_names),
    y=train_df["Label"]
)

class_weight_dict = dict(zip(range(len(class_names)), class_weights))
print(f"⚖️  Computed class weights: {class_weight_dict}")

# =====================================================================
# 5. IMAGE GENERATORS
# =====================================================================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_dataframe(
    train_df, x_col="Path", y_col="Label",
    target_size=(256, 256), class_mode="categorical",
    batch_size=16, shuffle=True, seed=SEED
)
val_gen = val_datagen.flow_from_dataframe(
    val_df, x_col="Path", y_col="Label",
    target_size=(256, 256), class_mode="categorical",
    batch_size=16, shuffle=False
)

# =====================================================================
# 6. ATTENTION BLOCK (CBAM)
# =====================================================================
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Conv2D, Dense, Multiply, Add, Activation, Concatenate

def cbam_block(input_tensor, ratio=8):
    """CBAM: Convolutional Block Attention Module (Channel + Spatial)."""
    channel = input_tensor.shape[-1]

    # ----- Channel Attention -----
    shared_dense_one = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)
    shared_dense_two = Dense(channel, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)

    avg_pool = GlobalAveragePooling2D()(input_tensor)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_dense_two(shared_dense_one(avg_pool))

    max_pool = GlobalMaxPooling2D()(input_tensor)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_dense_two(shared_dense_one(max_pool))

    channel_attention = Add()([avg_pool, max_pool])
    channel_attention = Activation('sigmoid')(channel_attention)
    channel_refined = Multiply()([input_tensor, channel_attention])

    # ----- Spatial Attention -----
    avg_pool = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(channel_refined)
    max_pool = tf.keras.layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(channel_refined)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    spatial_attention = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)
    refined_output = Multiply()([channel_refined, spatial_attention])

    return refined_output

# =====================================================================
# 7. MODEL DEFINITION (DenseNet121 + CBAM)
# =====================================================================
base_model = DenseNet121(include_top=False, weights="imagenet", input_shape=(256, 256, 3))
base_model.trainable = False  # Freeze base initially

x = base_model.output
x = cbam_block(x)
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(len(class_names), activation='softmax', dtype='float32')(x)  # ensure float32 for mixed precision

model = Model(inputs=base_model.input, outputs=outputs)
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name="auc")]
)

model.summary()

# =====================================================================
# 8. CALLBACKS
# =====================================================================
callbacks = [
    ModelCheckpoint(os.path.join(OUTPUT_DIR, "best_model_cbam.h5"),
                    monitor="val_auc", mode='max', save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=1),
    TensorBoard(log_dir=LOG_DIR, histogram_freq=1)
]

# =====================================================================
# 9. TRAINING STAGE 1 (FROZEN BASE)
# =====================================================================
print("\n🚀 Stage 1: Training classification head with frozen DenseNet...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=25,
    callbacks=callbacks,
    class_weight=class_weight_dict
)

# =====================================================================
# 10. FINE-TUNING STAGE 2
# =====================================================================
print("\n🔓 Stage 2: Fine-tuning last DenseNet layers + attention...")
base_model.trainable = True
for layer in base_model.layers[:-60]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name="auc")]
)

history_fine = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    callbacks=callbacks,
    class_weight=class_weight_dict
)

# =====================================================================
# 11. SAVE FINAL MODEL
# =====================================================================
final_path = os.path.join(OUTPUT_DIR, "final_model_cbam.h5")
model.save(final_path)
print(f"\n✅ Training completed. Final model saved to {final_path}")

# =====================================================================
# 12. PERFORMANCE VISUALIZATION
# =====================================================================
def plot_history(hist, title):
    plt.figure(figsize=(8, 4))
    plt.plot(hist.history['accuracy'], label='Train Acc')
    plt.plot(hist.history['val_accuracy'], label='Val Acc')
    plt.plot(hist.history['auc'], label='Train AUC')
    plt.plot(hist.history['val_auc'], label='Val AUC')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, f"{title.replace(' ', '_').lower()}.jpg"))
    plt.close()

plot_history(history, "Stage 1 Performance")
plot_history(history_fine, "Stage 2 Performance")

print("\n🏁 TRAINING SUMMARY")
print(f"Best Stage 1 Accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"Best Stage 1 AUC: {max(history.history['val_auc']):.4f}")
print(f"Best Stage 2 Accuracy: {max(history_fine.history['val_accuracy']):.4f}")
print(f"Best Stage 2 AUC: {max(history_fine.history['val_auc']):.4f}")

# =====================================================================
# 13. GRAD-CAM FOR EXPLAINABILITY
# =====================================================================
def generate_gradcam(model, img_array, layer_name='conv5_block16_concat'):
    grad_model = Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap[0], 0) / np.max(heatmap[0])
    return heatmap

print("\n🧠 Grad-CAM explainability ready (call generate_gradcam(...) for visualization)")

# =====================================================================
# 14. DETAILED VISUALIZATION 
# =====================================================================
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Load best model for evaluation
best_model = tf.keras.models.load_model(
    os.path.join(OUTPUT_DIR, "best_model_cbam.h5"),
    compile=False
)
best_model.compile(metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])

# Evaluate on test data
test_gen = val_datagen.flow_from_dataframe(
    test_df, x_col="Path", y_col="Label",
    target_size=(256, 256), class_mode="categorical",
    batch_size=16, shuffle=False
)

y_true = test_gen.classes
y_pred_proba = best_model.predict(test_gen)
y_pred = np.argmax(y_pred_proba, axis=1)

# =====================================================================
# 14.1 TRAINING HISTORY PLOTS (Loss, Accuracy, AUC)
# =====================================================================
def plot_training_metrics(histories, labels, save_prefix):
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 3, 1)
    for hist, label in zip(histories, labels):
        plt.plot(hist.history['accuracy'], label=f'{label} Train')
        plt.plot(hist.history['val_accuracy'], linestyle='--', label=f'{label} Val')
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    # Loss
    plt.subplot(1, 3, 2)
    for hist, label in zip(histories, labels):
        plt.plot(hist.history['loss'], label=f'{label} Train')
        plt.plot(hist.history['val_loss'], linestyle='--', label=f'{label} Val')
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # AUC
    plt.subplot(1, 3, 3)
    for hist, label in zip(histories, labels):
        plt.plot(hist.history['auc'], label=f'{label} Train')
        plt.plot(hist.history['val_auc'], linestyle='--', label=f'{label} Val')
    plt.title("AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, f"{save_prefix}_training_metrics.jpg")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 Saved training curves to {save_path}")

# Plot combined curves for Stage 1 and Stage 2
plot_training_metrics([history, history_fine], ["Stage 1", "Stage 2"], "combined")

# =====================================================================
# 14.2 CONFUSION MATRIX
# =====================================================================
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(6, 5))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.jpg")
plt.savefig(cm_path, dpi=300)
plt.close()
print(f"📈 Confusion matrix saved to {cm_path}")

# =====================================================================
# 14.3 ROC CURVES (MULTI-CLASS)
# =====================================================================
y_true_bin = label_binarize(y_true, classes=range(len(class_names)))

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(class_names)):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro/macro averages
fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves
plt.figure(figsize=(7, 6))
colors = cycle(["aqua", "darkorange", "cornflowerblue"])
for i, color in zip(range(len(class_names)), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.plot(fpr["micro"], tpr["micro"], color="deeppink", linestyle=":", lw=3,
         label=f"Micro-average (AUC = {roc_auc['micro']:.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curves")
plt.legend(loc="lower right")
roc_path = os.path.join(RESULTS_DIR, "roc_curves.jpg")
plt.savefig(roc_path, dpi=300)
plt.close()
print(f"📉 ROC curves saved to {roc_path}")

# =====================================================================
# 14.4 GRAD-CAM VISUALIZATION 
# =====================================================================
import cv2

def overlay_gradcam(img_path, heatmap, intensity=0.5, cmap=cv2.COLORMAP_JET):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    heatmap = cv2.resize(heatmap, (256, 256))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cmap)
    superimposed = cv2.addWeighted(heatmap, intensity, img, 1 - intensity, 0)
    return superimposed

# Example Grad-CAM visualization for one random test image
example_path = test_df.sample(1, random_state=SEED)["Path"].values[0]
img = tf.keras.preprocessing.image.load_img(example_path, target_size=(256, 256))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array / 255.0, axis=0)

heatmap = generate_gradcam(best_model, img_array)
overlay = overlay_gradcam(example_path, heatmap)

plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Grad-CAM Visualization Example")
gradcam_path = os.path.join(RESULTS_DIR, "gradcam_example.jpg")
plt.savefig(gradcam_path, dpi=300)
plt.close()
print(f"🔥 Grad-CAM visualization saved to {gradcam_path}")

# =====================================================================
# EXPORT METRİCS TO CSV
# =====================================================================
metrics_df = pd.DataFrame({
    "Stage1_Val_Acc": history.history['val_accuracy'],
    "Stage2_Val_Acc": history_fine.history['val_accuracy'],
    "Stage1_Val_AUC": history.history['val_auc'],
    "Stage2_Val_AUC": history_fine.history['val_auc']
})
metrics_df.to_csv(os.path.join(RESULTS_DIR, "training_summary.csv"), index=False)
print("📁 Training summary saved as CSV for article data tables.")
