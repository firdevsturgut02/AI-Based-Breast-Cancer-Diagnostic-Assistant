"""
model_factory.py
----------------
Model builder module for breast ultrasound classification.

Features:
- DenseNet121 (baseline)
- EfficientNetB3 (comparison)
- DenseNet121 + CBAM attention
- Fine-tuning utilities
- Architecture summary & logs auto-saved to results/
"""

# =====================================================================
# 1. IMPORTS
# =====================================================================
import os
import time
import tensorflow as tf

from tensorflow.keras.applications import DenseNet121, EfficientNetB3
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization,
    Conv2D, Multiply, Add, Activation, Reshape, GlobalMaxPooling2D
)
from tensorflow.keras.models import Model

# =====================================================================
# 2. PATHS
# =====================================================================
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# =====================================================================
# 3. CBAM ATTENTION BLOCK
# =====================================================================
def cbam_block(input_tensor, ratio=8):
    """Convolutional Block Attention Module (CBAM)."""
    channel = input_tensor.shape[-1]

    shared_dense_1 = Dense(channel // ratio, activation='relu', use_bias=False)
    shared_dense_2 = Dense(channel, activation='sigmoid', use_bias=False)

    # Channel Attention
    avg_pool = GlobalAveragePooling2D()(input_tensor)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_dense_2(shared_dense_1(avg_pool))

    max_pool = GlobalMaxPooling2D()(input_tensor)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_dense_2(shared_dense_1(max_pool))

    channel_attention = Add()([avg_pool, max_pool])
    channel_attention = Activation('sigmoid')(channel_attention)
    channel_refined = Multiply()([input_tensor, channel_attention])

    # Spatial Attention
    avg_spatial = tf.reduce_mean(channel_refined, axis=-1, keepdims=True)
    max_spatial = tf.reduce_max(channel_refined, axis=-1, keepdims=True)
    concat = tf.concat([avg_spatial, max_spatial], axis=-1)

    spatial_attention = Conv2D(
        1, kernel_size=7, padding='same', activation='sigmoid'
    )(concat)

    return Multiply()([channel_refined, spatial_attention])

# =====================================================================
# 4. DENSENET121 BASELINE
# =====================================================================
def build_densenet121(num_classes=3, input_shape=(256, 256, 3), weights='imagenet'):
    base_model = DenseNet121(
        include_top=False, weights=weights, input_shape=input_shape
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(base_model.input, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    print("✅ DenseNet121 baseline built.")
    return model

# =====================================================================
# 5. EFFICIENTNETB3
# =====================================================================
def build_efficientnetb3(num_classes=3, input_shape=(256, 256, 3), weights='imagenet'):
    base_model = EfficientNetB3(
        include_top=False, weights=weights, input_shape=input_shape
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(base_model.input, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    print("✅ EfficientNetB3 built.")
    return model

# =====================================================================
# 6. DENSENET121 + CBAM
# =====================================================================
def build_densenet_cbam(num_classes=3, input_shape=(256, 256, 3), weights='imagenet'):
    base_model = DenseNet121(
        include_top=False, weights=weights, input_shape=input_shape
    )
    base_model.trainable = False

    x = base_model.output
    x = cbam_block(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(base_model.input, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    print("✅ DenseNet121 + CBAM built.")
    return model

# =====================================================================
# 7. FINE-TUNING UTILITY (LOGGED)
# =====================================================================
def fine_tune_model(model, num_layers_to_unfreeze=50):
    start = time.time()

    base_model = None
    for layer in model.layers:
        if "densenet" in layer.name.lower() or "efficientnet" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        base_model = model

    base_model.trainable = True
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    trainable_layers = sum(l.trainable for l in base_model.layers)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    elapsed = time.time() - start
    device = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"

    log_text = (
        f"Fine-tuning enabled\n"
        f"Unfrozen layers: {num_layers_to_unfreeze}\n"
        f"Trainable layers: {trainable_layers}\n"
        f"Device: {device}\n"
        f"Time: {elapsed:.2f}s\n"
    )

    log_path = os.path.join(RESULTS_DIR, "fine_tuning_log.txt")
    with open(log_path, "a") as f:
        f.write(log_text + "\n")

    print("🔓 Fine-tuning configured.")
    return model

# =====================================================================
# 8. MODEL FACTORY ENTRY POINT
# =====================================================================
def get_model(
    model_name="densenet121",
    num_classes=3,
    input_shape=(256, 256, 3),
    weights="imagenet",
    export_summary=True
):
    model_map = {
        "densenet121": build_densenet121,
        "efficientnetb3": build_efficientnetb3,
        "densenet_cbam": build_densenet_cbam
    }

    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}")

    model = model_map[model_name](
        num_classes=num_classes,
        input_shape=input_shape,
        weights=weights
    )

    if export_summary:
        summary_path = os.path.join(
            RESULTS_DIR, f"{model_name}_summary.txt"
        )
        with open(summary_path, "w") as f:
            model.summary(print_fn=lambda x: f.write(x + "\n"))
        print(f"🧾 Model summary saved to {summary_path}")

    return model
