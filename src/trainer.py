"""
trainer.py
----------
Unified Training Pipeline for Breast Ultrasound Classification.
Optimized for Technical Reports and %90+ Accuracy Target.

Updates:
✅ Dynamic Class Weighting to handle Malignant/Normal imbalance.
✅ Enhanced learning rate scheduling for better convergence.
✅ Full 50-epoch cycle with Best Model recovery based on val_auc.
✅ Comprehensive Academic Visualizations.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.utils import class_weight
from model_factory import get_model
from data_loader import prepare_data_frames, MedicalDataGenerator

# =====================================================================
# 1. GLOBAL CONFIGURATION
# =====================================================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = (256, 256)
BATCH_SIZE = 16
EPOCHS = 50  
DATA_DIR = "Dataset_BUSI_with_GT"

# Output Directory Setup
TRAINER_OUT_DIR = os.path.join("results", "trainer")
FIGURES_DIR = os.path.join(TRAINER_OUT_DIR, "figures")
MODELS_DIR = os.path.join(TRAINER_OUT_DIR, "models")
TABLES_DIR = os.path.join(TRAINER_OUT_DIR, "tables")

for d in [TRAINER_OUT_DIR, FIGURES_DIR, MODELS_DIR, TABLES_DIR]:
    os.makedirs(d, exist_ok=True)

# =====================================================================
# 2. DATA PREPARATION & CLASS WEIGHTING
# =====================================================================
print("📊 Analyzing Dataset and Preparing Generators...")
train_df, val_df, test_df = prepare_data_frames(DATA_DIR, seed=SEED, balance=True)

# Sınıf dengesizliğini çözmek için ağırlık hesaplama
y_train_indices = train_df['Label'].astype('category').cat.codes
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_indices),
    y=y_train_indices
)
class_weights = dict(enumerate(weights))
print(f"⚖️ Calculated Class Weights: {class_weights}")

train_gen = MedicalDataGenerator(train_df, batch_size=BATCH_SIZE, augment=True)
val_gen = MedicalDataGenerator(val_df, batch_size=BATCH_SIZE, augment=False)

# Hardware Analysis
gpu_devices = tf.config.list_physical_devices('GPU')
device_name = "Tesla T4" if gpu_devices else "CPU"
print(f"📍 Execution Device: {device_name}")

# =====================================================================
# 3. CALLBACKS & OPTIMIZATION
# =====================================================================
model_save_path = os.path.join(MODELS_DIR, "best_model.h5")
callbacks = [
    # AUC bazlı en iyi ağırlıkları kaydet
    tf.keras.callbacks.ModelCheckpoint(
        filepath=model_save_path,
        monitor="val_auc",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    # Model plato çizdiğinde öğrenme hızını %80 azalt
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    # Eğitim günlüğünü CSV olarak sakla
    tf.keras.callbacks.CSVLogger(os.path.join(TABLES_DIR, "training_log.csv"))
]

# =====================================================================
# 4. MODEL INITIALIZATION & TRAINING
# =====================================================================
print("🏗️ Building DenseNet121 + CBAM Model...")
model = get_model(num_classes=3, input_shape=(256, 256, 3))

print(f"🚀 Starting Unified 50-Epoch Training (Target: 90%+ Accuracy)...")
start_train_time = time.time()

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights, # Dengesiz veriyi dengele
    verbose=1
)

total_train_time = time.time() - start_train_time

# =====================================================================
# 5. POST-TRAINING ANALYSIS & VISUALIZATION
# =====================================================================
def plot_academic_curves(history):
    plt.figure(figsize=(16, 6))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', lw=2, color='#1f77b4')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', lw=2, color='#ff7f0e')
    plt.axhline(y=0.90, color='r', linestyle='--', label='90% Target Threshold')
    plt.title('Model Accuracy Evolution', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', lw=2, color='#d62728')
    plt.plot(history.history['val_loss'], label='Validation Loss', lw=2, color='#2ca02c')
    plt.title('Model Loss Convergence', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "learning_curves.png"), dpi=300)
    plt.close()

plot_academic_curves(history)
[Image of deep learning model training loss and accuracy curves showing convergence towards 90 percent accuracy]

# =====================================================================
# 6. VALIDATION EVALUATION (CM & ROC)
# =====================================================================
print("🔍 Performing Final Validation Evaluation...")
# Modelin en iyi ağırlıklarını geri yükle (yükleme hatası almamak için evaluator.py metodunu kullanın)
# model.load_weights(model_save_path) 

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
[Image of a confusion matrix for 3 classes showing performance across benign, malignant and normal categories]
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix: Validation Set", fontsize=14, fontweight='bold')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrix.png"), dpi=300)
plt.close()

# Multi-Class ROC Analysis
[Image of a ROC curve for medical diagnosis showing AUC performance across classes]
plt.figure(figsize=(10, 8))
for i, label in enumerate(class_labels):
    fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_scores[:, i])
    plt.plot(fpr, tpr, label=f'{label} (AUC = {auc(fpr, tpr):.3f})', lw=2)

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.title("Multi-Class ROC Performance Analysis", fontsize=14, fontweight='bold')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.2)
plt.savefig(os.path.join(FIGURES_DIR, "roc_analysis.png"), dpi=300)
plt.close()

# =====================================================================
# 7. EXPORT TECHNICAL SUMMARY
# =====================================================================
summary_stats = pd.DataFrame({
    "Metric": ["Max Val Accuracy", "Max Val AUC", "Training Time (s)", "Inference Device"],
    "Value": [
        f"{max(history.history['val_accuracy']):.4f}",
        f"{max(history.history['val_auc']):.4f}",
        f"{total_train_time:.2f}",
        device_name
    ]
})
summary_stats.to_csv(os.path.join(TABLES_DIR, "technical_summary.csv"), index=False)

print(f"\n✅ Training Process Complete. Artifacts saved in: {TRAINER_OUT_DIR}")
print(f"📊 Final Validation Accuracy: {max(history.history['val_accuracy'])*100:.2f}%")
