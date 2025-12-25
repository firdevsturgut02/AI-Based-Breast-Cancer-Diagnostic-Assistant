"""
trainer.py
----------
Unified Training Pipeline for Breast Ultrasound Classification.
Optimized for Technical Reports and %90+ Accuracy Target.

Key Features:
✅ 50-Epoch Unstoppable Training (Early Stopping Removed).
✅ Hardware Performance & Inference Latency Metrics.
✅ High-Resolution Confusion Matrix & ROC Curves.
✅ Scientific Technical Summary (CSV).
"""

# =====================================================================
# 1. IMPORTS
# =====================================================================
import os
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from model_factory import get_model
from data_loader import prepare_data_frames, MedicalDataGenerator

# =====================================================================
# 2. GLOBAL CONFIGURATION
# =====================================================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = (256, 256)
BATCH_SIZE = 16
EPOCHS = 50  
NUM_CLASSES = 3
DATA_DIR = "Dataset_BUSI_with_GT"

# Output Directory Setup
TRAINER_OUT_DIR = os.path.join("results", "trainer")
FIGURES_DIR = os.path.join(TRAINER_OUT_DIR, "figures")
MODELS_DIR = os.path.join(TRAINER_OUT_DIR, "models")
TABLES_DIR = os.path.join(TRAINER_OUT_DIR, "tables")

for d in [TRAINER_OUT_DIR, FIGURES_DIR, MODELS_DIR, TABLES_DIR]:
    os.makedirs(d, exist_ok=True)

# =====================================================================
# 3. DATA PREPARATION
# =====================================================================
print("📊 Analyzing Dataset and Applying Class Balancing...")
train_df, val_df, test_df = prepare_data_frames(DATA_DIR, seed=SEED, balance=True)

print("⚙️ Initializing Scientific Data Generators...")
train_gen = MedicalDataGenerator(train_df, batch_size=BATCH_SIZE, augment=True)
val_gen = MedicalDataGenerator(val_df, batch_size=BATCH_SIZE, augment=False)

# Hardware Info
gpu_devices = tf.config.list_physical_devices('GPU')
device_name = tf.test.gpu_device_name() if gpu_devices else "CPU"
print(f"📍 Execution Device: {device_name}")

# =====================================================================
# 4. CALLBACKS (EarlyStopping Removed)
# =====================================================================
model_save_path = os.path.join(MODELS_DIR, "best_model.h5")
callbacks = [
    # AUC bazlı en iyi modeli kaydet (Her zaman en iyi performansı yakalar)
    tf.keras.callbacks.ModelCheckpoint(
        filepath=model_save_path,
        monitor="val_auc",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    # Hassas ayar (Fine-tuning) için öğrenme oranını otomatik ayarla
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# =====================================================================
# 5. TRAINING SESSION
# =====================================================================
print("🏗️ Building DenseNet121 + CBAM Model...")
model = get_model(num_classes=NUM_CLASSES, input_shape=(256, 256, 3))

print(f"🚀 Starting Unified 50-Epoch Training (Target: 90%+ Accuracy)...")
start_train_time = time.time()
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)
total_train_time = time.time() - start_train_time

# =====================================================================
# 6. PERFORMANCE & HARDWARE ANALYSIS
# =====================================================================
def measure_inference(model, iterations=100):
    dummy_input = np.random.rand(1, 256, 256, 3).astype(np.float32)
    for _ in range(10): _ = model.predict(dummy_input, verbose=0) # Warmup
    start = time.time()
    for _ in range(iterations): _ = model.predict(dummy_input, verbose=0)
    return ((time.time() - start) / iterations) * 1000

avg_latency = measure_inference(model)
print(f"⏱️ Inference Latency: {avg_latency:.2f} ms")

# =====================================================================
# 7. SCIENTIFIC VISUALIZATION (Accuracy, Loss, CM, ROC)
# =====================================================================
# Accuracy & Loss Curves

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', lw=2)
plt.plot(history.history['val_accuracy'], label='Val Accuracy', lw=2)
plt.axhline(y=0.90, color='r', linestyle='--', label='90% Target')
plt.title('Accuracy Evolution')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', lw=2)
plt.plot(history.history['val_loss'], label='Val Loss', lw=2)
plt.title('Loss Convergence')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(FIGURES_DIR, "learning_curves.png"), dpi=300); plt.close()

# Evaluate on Validation Set
y_true, y_scores = [], []
for i in range(len(val_gen)):
    x, y = val_gen[i]
    y_true.append(np.argmax(y, axis=1))
    y_scores.append(model.predict(x, verbose=0))

y_true = np.concatenate(y_true)
y_scores = np.concatenate(y_scores)
y_pred = np.argmax(y_scores, axis=1)
class_labels = sorted(train_df["Label"].unique())

# Confusion Matrix

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues", 
            xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix (Validation Set)")
plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrix.png"), dpi=300); plt.close()

# Multi-class ROC
plt.figure(figsize=(9, 7))
for i, label in enumerate(class_labels):
    fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_scores[:, i])
    plt.plot(fpr, tpr, label=f'{label} (AUC = {auc(fpr, tpr):.3f})', lw=2)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.title("Multi-Class ROC Curves"); plt.legend(); plt.grid(True, alpha=0.2)
plt.savefig(os.path.join(FIGURES_DIR, "roc_analysis.png"), dpi=300); plt.close()

# =====================================================================
# 8. TECHNICAL SUMMARY EXPORT
# =====================================================================
summary_stats = pd.DataFrame({
    "Parameter": ["Best Val Accuracy", "Inference Latency (ms)", "Total Training Time (s)", "Device"],
    "Value": [max(history.history['val_accuracy']), avg_latency, total_train_time, device_name]
})
summary_stats.to_csv(os.path.join(TABLES_DIR, "technical_summary.csv"), index=False)

print(f"\n✅ Training Process Complete.")
print(f"💾 Best Model Saved: {model_save_path}")
print(f"📁 Scientific Artifacts: {TRAINER_OUT_DIR}")
