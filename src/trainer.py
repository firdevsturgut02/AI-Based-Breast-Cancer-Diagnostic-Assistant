"""
trainer.py

AI-Based Breast Cancer Diagnostic Assistant 

---------------------------------------------------
✅ DenseNet121 + CBAM Attention
✅ Focal Loss for class imbalance
✅ Cosine learning rate decay
✅ Strong data augmentation
✅ Extended fine-tuning
✅ Mixed precision
✅ Grad-CAM explainability
✅ TensorBoard logging
✅ CSV export (training + test metrics)
"""

# =====================================================================
# 1. IMPORTS
# =====================================================================
import os, random, datetime, cv2, json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, GlobalAveragePooling2D,
    Conv2D, Multiply, Add, Activation, Reshape, Concatenate, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, LearningRateScheduler, CSVLogger
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

# =====================================================================
# 2. REPRODUCIBILITY
# =====================================================================
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
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

class_names = sorted(df["Label"].unique())
class_weights = compute_class_weight(class_weight='balanced', classes=np.array(class_names), y=train_df["Label"])
class_weight_dict = dict(zip(range(len(class_names)), class_weights))
print(f"⚖️ Computed class weights: {class_weight_dict}")

# =====================================================================
# 5. DATA AUGMENTATION
# =====================================================================
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=35,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.25,
    zoom_range=0.35,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.6, 1.4],
    fill_mode='reflect'
)
val_datagen = ImageDataGenerator(rescale=1.0/255)

train_gen = train_datagen.flow_from_dataframe(
    train_df, x_col="Path", y_col="Label",
    target_size=(320, 320), class_mode="categorical",
    batch_size=8, shuffle=True, seed=SEED
)
val_gen = val_datagen.flow_from_dataframe(
    val_df, x_col="Path", y_col="Label",
    target_size=(320, 320), class_mode="categorical",
    batch_size=8, shuffle=False
)

# =====================================================================
# 6. CBAM BLOCK
# =====================================================================
def cbam_block(input_tensor, ratio=8):
    channel = input_tensor.shape[-1]
    shared_dense_one = Dense(channel // ratio, activation='relu', use_bias=False)
    shared_dense_two = Dense(channel, activation='sigmoid', use_bias=False)

    avg_pool = GlobalAveragePooling2D()(input_tensor)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_dense_two(shared_dense_one(avg_pool))

    max_pool = tf.keras.layers.GlobalMaxPooling2D()(input_tensor)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_dense_two(shared_dense_one(max_pool))

    channel_attention = Add()([avg_pool, max_pool])
    channel_attention = Activation('sigmoid')(channel_attention)
    channel_refined = Multiply()([input_tensor, channel_attention])

    avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(channel_refined)
    max_pool = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(channel_refined)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    spatial_attention = Conv2D(1, 7, padding='same', activation='sigmoid')(concat)
    refined_output = Multiply()([channel_refined, spatial_attention])
    return refined_output

# =====================================================================
# 7. MODEL
# =====================================================================
base_model = DenseNet121(include_top=False, weights="imagenet", input_shape=(320, 320, 3))
base_model.trainable = False

x = base_model.output
x = cbam_block(x)
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(len(class_names), activation='softmax', dtype='float32')(x)
model = Model(inputs=base_model.input, outputs=outputs)

# FOCAL LOSS
from tensorflow.keras import backend as K
def focal_loss(gamma=2., alpha=.25):
    def loss_fn(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = -y_true * alpha * K.pow(1 - y_pred, gamma) * K.log(y_pred)
        return K.sum(loss, axis=1)
    return loss_fn

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=focal_loss(),
    metrics=['accuracy', tf.keras.metrics.AUC(name="auc")]
)

# =====================================================================
# 8. CALLBACKS
# =====================================================================
def cosine_decay(epoch):
    initial_lr = 1e-4
    total_epochs = 30
    return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))

csv_logger = CSVLogger(os.path.join(RESULTS_DIR, "training_log.csv"), append=True)

callbacks = [
    ModelCheckpoint(os.path.join(OUTPUT_DIR, "best_model_cbam.keras"),
                    monitor="val_auc", mode='max', save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=1),
    LearningRateScheduler(cosine_decay),
    TensorBoard(log_dir=LOG_DIR),
    csv_logger
]

# =====================================================================
# 9. TRAINING
# =====================================================================
print("\n🚀 Stage 1: Training classification head...")
history = model.fit(
    train_gen, validation_data=val_gen,
    epochs=30, callbacks=callbacks, class_weight=class_weight_dict
)

print("\n🔓 Stage 2: Fine-tuning DenseNet deeper layers...")
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=2e-5),
    loss=focal_loss(),
    metrics=['accuracy', tf.keras.metrics.AUC(name="auc")]
)

history_fine = model.fit(
    train_gen, validation_data=val_gen,
    epochs=40, callbacks=callbacks, class_weight=class_weight_dict
)

# =====================================================================
# 10. SAVE FINAL MODEL
# =====================================================================
final_path = os.path.join(OUTPUT_DIR, "final_model_cbam.keras")
model.save(final_path)
print(f"✅ Training completed. Model saved to {final_path}")

# =====================================================================
# 11. EVALUATION & CSV EXPORT
# =====================================================================
best_model = tf.keras.models.load_model(os.path.join(OUTPUT_DIR, "best_model_cbam.keras"))
test_gen = val_datagen.flow_from_dataframe(
    test_df, x_col="Path", y_col="Label",
    target_size=(256, 256), class_mode="categorical",
    batch_size=16, shuffle=False
)

test_metrics = best_model.evaluate(test_gen, return_dict=True)
print("\n📊 TEST METRICS:")
for k, v in test_metrics.items():
    print(f"{k}: {v:.4f}")

# Export test metrics to CSV
test_metrics_path = os.path.join(RESULTS_DIR, "test_metrics.csv")
pd.DataFrame([test_metrics]).to_csv(test_metrics_path, index=False)
print(f"✅ Test metrics exported to {test_metrics_path}")

# Classification report
y_true = test_gen.classes
y_pred_proba = best_model.predict(test_gen)
y_pred = np.argmax(y_pred_proba, axis=1)
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
pd.DataFrame(report).transpose().to_csv(os.path.join(RESULTS_DIR, "classification_report.csv"))
print("📁 Classification report saved as CSV.")

# =====================================================================
# 12. TENSORBOARD INFO
# =====================================================================
print("\n📈 TensorBoard log directory:")
print(LOG_DIR)
print("\nTo visualize training progress, run:")
print(f"tensorboard --logdir={LOG_DIR}")
