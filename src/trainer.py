"""
trainer.py
----------
Unified Training Pipeline for Breast Ultrasound Classification.
Optimized for Technical Reports and Academic Standards.

Key Features:
✅ Single-stage 50-epoch training (full network optimization).
✅ Hardware Performance Metrics (Inference Time & GPU Analysis).
✅ Automated Results Organization in results/trainer/.
✅ Comprehensive Visualization: Confusion Matrix, ROC Curves, Accuracy/Loss.
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
print("📊 Preparing Dataframes and Analyzing Distribution...")
train_df, val_df, test_df = prepare_data_frames(DATA_DIR, seed=SEED, balance=True)

print("⚙️ Initializing Medical Data Generators...")
train_gen = MedicalDataGenerator(train_df, batch_size=BATCH_SIZE, augment=True)
val_gen = MedicalDataGenerator(val_df, batch_size=BATCH_SIZE, augment=False)

# =====================================================================
# 4. HARDWARE & INFERENCE ANALYSIS PRE-TRAIN
# =====================================================================
gpu_devices = tf.config.list_physical_devices('GPU')
device_name = tf.test.gpu_device_name() if gpu_devices else "CPU"
print(f"📍 Execution Device: {device_name}")

# =====================================================================
# 5. CALLBACKS
# =====================================================================
model_save_path = os.path.join(MODELS_DIR, "best_model.h5") #
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=model_save_path,
        monitor="val_auc",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10, 
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# =====================================================================
# 6. MODEL INITIALIZATION & TRAINING
# =====================================================================
print("🏗️ Building DenseNet121 + CBAM Architecture...")
model = get_model(num_classes=NUM_CLASSES, input_shape=(256, 256, 3))

print(f"🚀 Starting Unified Training Session ({EPOCHS} Epochs)...")
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
# 7. INFERENCE TIME ANALYSIS (POST-TRAIN)
# =====================================================================
def measure_inference(model, iterations=100):
    dummy_input = np.random.rand(1, 256, 256, 3).astype(np.float32)
    # Warm-up
    for _ in range(10): _ = model.predict(dummy_input, verbose=0)
    
    start = time.time()
    for _ in range(iterations): _ = model.predict(dummy_input, verbose=0)
    avg_latency = ((time.time() - start) / iterations) * 1000 # ms
    return avg_latency

avg_latency = measure_inference(model)
print(f"⏱️ Average Inference Latency: {avg_latency:.2f} ms")

# =====================================================================
# 8. VISUALIZATION: ACCURACY & LOSS
# =====================================================================
def plot_learning_curves(history):
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', lw=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', lw=2)
    plt.title('Model Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', lw=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', lw=2)
    plt.title('Model Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "learning_curves.png"), dpi=300)
    plt.close()

plot_learning_curves(history)

# =====================================================================
# 9. PERFORMANCE EVALUATION: CM & ROC
# =====================================================================
print("🔍 Performing Post-Training Evaluation...")
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
plt.title("Confusion Matrix")
plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrix.png"), dpi=300); plt.close()

# ROC Curves
plt.figure(figsize=(9, 7))
for i, label in enumerate(class_labels):
    fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_scores[:, i])
    plt.plot(fpr, tpr, label=f'{label} (AUC = {auc(fpr, tpr):.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title("Multi-Class ROC Analysis"); plt.legend(); plt.grid(True, alpha=0.2)
plt.savefig(os.path.join(FIGURES_DIR, "roc_analysis.png"), dpi=300); plt.close()

# =====================================================================
# 10. TECHNICAL SUMMARY EXPORT
# =====================================================================
summary_stats = pd.DataFrame({
    "Parameter": ["Best Val Accuracy", "Inference Latency (ms)", "Total Training Time (s)", "Device"],
    "Value": [max(history.history['val_accuracy']), avg_latency, total_train_time, device_name]
})
summary_stats.to_csv(os.path.join(TABLES_DIR, "technical_summary.csv"), index=False)

print(f"\n✅ Training Complete. Model saved to: {model_save_path}")
print(f"📁 Artifacts saved in: {TRAINER_OUT_DIR}")
