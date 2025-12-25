"""
trainer.py
-----------
Unified Scientific Training Pipeline for Breast Ultrasound Image Classification.

Version: 2.0 (Academic Edition)
Target: >90% Accuracy with High Reproducibility and Technical Traceability.

Key Features:
    ✅ 50-Epoch Continuous Training (No Early Stopping)
    ✅ GPU/CPU Hardware Benchmark & Inference Latency Measurement
    ✅ Publication-Ready Figures (Confusion Matrix & ROC)
    ✅ Technical Summary Export (CSV)
    ✅ Deterministic, Modular, and PEP8-Compliant

Author: [Your Name]
Affiliation: [Your Institution]
Date: [Auto-generated at runtime]
"""

# =====================================================================
# 1. IMPORTS
# =====================================================================
import os
import time
import datetime
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import Tuple, List, Dict

from model_factory import get_model
from data_loader import prepare_data_frames, MedicalDataGenerator


# =====================================================================
# 2. CONFIGURATION & LOGGING SETUP
# =====================================================================
SEED: int = 42
IMG_SIZE: Tuple[int, int] = (256, 256)
BATCH_SIZE: int = 16
EPOCHS: int = 50
NUM_CLASSES: int = 3
DATA_DIR: str = "Dataset_BUSI_with_GT"

# Output directories
TRAINER_OUT_DIR = os.path.join("results", "trainer")
FIGURES_DIR = os.path.join(TRAINER_OUT_DIR, "figures")
MODELS_DIR = os.path.join(TRAINER_OUT_DIR, "models")
TABLES_DIR = os.path.join(TRAINER_OUT_DIR, "tables")

for d in [TRAINER_OUT_DIR, FIGURES_DIR, MODELS_DIR, TABLES_DIR]:
    os.makedirs(d, exist_ok=True)

# Set seeds for reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Configure logging
logging.basicConfig(
    filename=os.path.join(TRAINER_OUT_DIR, "training_log.txt"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

logging.info("=== Unified Breast Ultrasound Classifier Training Initialized ===")
logging.info(f"Seed: {SEED}, Image Size: {IMG_SIZE}, Batch Size: {BATCH_SIZE}, Epochs: {EPOCHS}")


# =====================================================================
# 3. DATA PREPARATION
# =====================================================================
logging.info("📊 Preparing dataset and applying class balancing...")
train_df, val_df, test_df = prepare_data_frames(DATA_DIR, seed=SEED, balance=True)

logging.info("⚙️ Initializing scientific data generators...")
train_gen = MedicalDataGenerator(train_df, batch_size=BATCH_SIZE, augment=True)
val_gen = MedicalDataGenerator(val_df, batch_size=BATCH_SIZE, augment=False)

gpu_devices = tf.config.list_physical_devices('GPU')
device_name = tf.test.gpu_device_name() if gpu_devices else "CPU"
logging.info(f"📍 Execution Device: {device_name}")


# =====================================================================
# 4. CALLBACKS (No Early Stopping)
# =====================================================================
model_save_path = os.path.join(MODELS_DIR, "best_model.h5")

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=model_save_path,
        monitor="val_auc",
        save_best_only=True,
        mode="max",
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
# 5. MODEL TRAINING
# =====================================================================
def train_model() -> tf.keras.callbacks.History:
    """Train the model with full logging and return training history."""
    logging.info("🏗️ Building DenseNet121 + CBAM Architecture...")
    model = get_model(num_classes=NUM_CLASSES, input_shape=(256, 256, 3))
    model.summary(print_fn=logging.info)

    logging.info("🚀 Starting 50-Epoch Unified Training (Goal: >90% Accuracy)")
    start_train_time = time.time()
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    total_time = time.time() - start_train_time
    logging.info(f"✅ Training completed in {total_time / 60:.2f} minutes.")
    return model, history, total_time


# =====================================================================
# 6. PERFORMANCE & HARDWARE ANALYSIS
# =====================================================================
def measure_inference_latency(model: tf.keras.Model, iterations: int = 100) -> float:
    """Measure average inference latency (ms)."""
    dummy_input = np.random.rand(1, 256, 256, 3).astype(np.float32)
    for _ in range(10):
        _ = model.predict(dummy_input, verbose=0)  # Warmup
    start = time.time()
    for _ in range(iterations):
        _ = model.predict(dummy_input, verbose=0)
    avg_latency = ((time.time() - start) / iterations) * 1000
    return avg_latency


# =====================================================================
# 7. VISUALIZATION & ANALYTICS
# =====================================================================
def plot_learning_curves(history: tf.keras.callbacks.History):
    """Plot accuracy and loss evolution."""
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

    plt.savefig(os.path.join(FIGURES_DIR, "learning_curves.png"), dpi=300)
    plt.close()


def evaluate_and_plot(model: tf.keras.Model):
    """Generate confusion matrix and ROC analysis."""
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
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d",
                cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title("Confusion Matrix (Validation Set)")
    plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrix.png"), dpi=300)
    plt.close()

    # Multi-class ROC
    plt.figure(figsize=(9, 7))
    for i, label in enumerate(class_labels):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_scores[:, i])
        plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {auc(fpr, tpr):.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.title("Multi-Class ROC Curves")
    plt.legend(); plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(FIGURES_DIR, "roc_analysis.png"), dpi=300)
    plt.close()

    return y_true, y_pred


# =====================================================================
# 8. TECHNICAL SUMMARY EXPORT
# =====================================================================
def export_summary(history, latency, total_time, device):
    """Export concise technical metrics for publication or benchmarking."""
    summary_stats = pd.DataFrame({
        "Parameter": ["Best Val Accuracy", "Inference Latency (ms)", "Total Training Time (s)", "Device"],
        "Value": [max(history.history['val_accuracy']), latency, total_time, device]
    })
    summary_stats.to_csv(os.path.join(TABLES_DIR, "technical_summary.csv"), index=False)
    logging.info("📁 Technical summary exported successfully.")


# =====================================================================
# 9. MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":
    model, history, total_time = train_model()
    latency = measure_inference_latency(model)
    logging.info(f"⏱️ Inference Latency: {latency:.2f} ms")

    plot_learning_curves(history)
    evaluate_and_plot(model)
    export_summary(history, latency, total_time, device_name)

    logging.info(f"💾 Best Model Saved: {model_save_path}")
    logging.info(f"📁 All Artifacts Stored in: {TRAINER_OUT_DIR}")
    logging.info("🎓 Training process completed successfully (Academic Standard).")
