"""
trainer.py
----------
Training pipeline for breast ultrasound image classification.

This script:
- Loads dataset
- Trains selected CNN architecture
- Performs fine-tuning
- Saves all models, figures, tables, and logs under results/
- Produces publication-ready outputs (ROC, CM, Grad-CAM)

"""

# =====================================================================
# 1. IMPORTS
# =====================================================================
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns

from model_factory import get_model, fine_tune_model


# =====================================================================
# 2. GLOBAL CONFIGURATION
# =====================================================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = (256, 256)
BATCH_SIZE = 16
EPOCHS_STAGE_1 = 25
EPOCHS_STAGE_2 = 20
NUM_CLASSES = 3

MODEL_NAME = "densenet_cbam"


# =====================================================================
# 3. DIRECTORY STRUCTURE (ALL OUTPUTS UNDER results/)
# =====================================================================
DATA_DIR = "Dataset_BUSI_with_GT"

RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
GRADCAM_DIR = os.path.join(RESULTS_DIR, "gradcam")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
LOG_DIR = os.path.join(
    RESULTS_DIR,
    "logs",
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)

for d in [RESULTS_DIR, FIGURES_DIR, GRADCAM_DIR, MODELS_DIR, TABLES_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)


# =====================================================================
# 4. DATA GENERATORS
# =====================================================================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    seed=SEED
)

val_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    seed=SEED
)


# =====================================================================
# 5. CALLBACKS
# =====================================================================
callbacks = [
    ModelCheckpoint(
        filepath=os.path.join(MODELS_DIR, "best_model.h5"),
        monitor="val_auc",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=4,
        min_lr=1e-6
    ),
    TensorBoard(log_dir=LOG_DIR)
]


# =====================================================================
# 6. MODEL BUILDING
# =====================================================================
model = get_model(
    model_name=MODEL_NAME,
    num_classes=NUM_CLASSES,
    input_shape=(256, 256, 3),
    export_summary=True
)


# =====================================================================
# 7. STAGE 1 TRAINING (FEATURE EXTRACTION)
# =====================================================================
history_stage_1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_STAGE_1,
    callbacks=callbacks
)


# =====================================================================
# 8. STAGE 2 TRAINING (FINE-TUNING)
# =====================================================================
model = fine_tune_model(model, num_layers_to_unfreeze=50)

history_stage_2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_STAGE_2,
    callbacks=callbacks
)

model.save(os.path.join(MODELS_DIR, "final_model.h5"))


# =====================================================================
# 9. TRAINING METRICS PLOT
# =====================================================================
def plot_training(history1, history2):
    acc = history1.history["accuracy"] + history2.history["accuracy"]
    val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
    loss = history1.history["loss"] + history2.history["loss"]
    val_loss = history1.history["val_loss"] + history2.history["val_loss"]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title("Accuracy")
    plt.legend(["Train", "Validation"])

    plt.subplot(1, 2, 2)
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title("Loss")
    plt.legend(["Train", "Validation"])

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "training_metrics.jpg"), dpi=300)
    plt.close()

plot_training(history_stage_1, history_stage_2)


# =====================================================================
# 10. CONFUSION MATRIX & ROC
# =====================================================================
val_gen.reset()
preds = model.predict(val_gen)
y_pred = np.argmax(preds, axis=1)
y_true = val_gen.classes

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrix.jpg"), dpi=300)
plt.close()

# ROC CURVES
plt.figure(figsize=(7, 6))
for i in range(NUM_CLASSES):
    fpr, tpr, _ = roc_curve((y_true == i).astype(int), preds[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {i} (AUC={roc_auc:.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.savefig(os.path.join(FIGURES_DIR, "roc_curves.jpg"), dpi=300)
plt.close()


# =====================================================================
# 11. METRICS TABLE EXPORT
# =====================================================================
metrics_df = pd.DataFrame({
    "Metric": ["Final Train Accuracy", "Final Validation Accuracy"],
    "Value": [
        history_stage_2.history["accuracy"][-1],
        history_stage_2.history["val_accuracy"][-1]
    ]
})

metrics_df.to_csv(
    os.path.join(TABLES_DIR, "training_summary.csv"),
    index=False
)

print("\n✅ Training completed successfully.")
print("📁 All results saved under the 'results/' directory.")
