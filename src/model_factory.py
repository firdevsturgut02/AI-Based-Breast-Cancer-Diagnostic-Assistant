"""
model_factory.py
----------------
Model builder module for breast ultrasound classification.

Updated Version:
✅ Compatible with trainer.py (accepts model_name argument)
✅ Uses only the best-performing architecture — DenseNet121 + CBAM
✅ Includes CBAM (Convolutional Block Attention Module)
✅ Supports fine-tuning
✅ Automatically saves model summaries and logs to results/
"""

# =====================================================================
# 1. IMPORTS
# =====================================================================
import os
import time
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
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
    """
    Convolutional Block Attention Module (CBAM)
    Applies both Channel and Spatial Attention mechanisms.
    """
    channel = input_tensor.shape[-1]

    # Shared MLP for channel attention
    shared_dense_1 = Dense(channel // ratio, activation='relu', use_bias=False)
    shared_dense_2 = Dense(channel, activation='sigmoid', use_bias=False)

    # ----- Channel Attention -----
    avg_pool = GlobalAveragePooling2D()(input_tensor)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_dense_2(shared_dense_1(avg_pool))

    max_pool = GlobalMaxPooling2D()(input_tensor)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_dense_2(shared_dense_1(max_pool))

    channel_attention = Add()([avg_pool, max_pool])
    channel_attention = Activation('sigmoid')(channel_attention)
    channel_refined = Multiply()([input_tensor, channel_attention])

    # ----- Spatial Attention -----
    avg_spatial = tf.reduce_mean(channel_refined, axis=-1, keepdims=True)
    max_spatial = tf.reduce_max(channel_refined, axis=-1, keepdims=True)
    concat = tf.concat([avg_spatial, max_spatial], axis=-1)

    spatial_attention = Conv2D(
        1, kernel_size=7, padding='same', activation='sigmoid'
    )(concat)

    refined_output = Multiply()([channel_refined, spatial_attention])
    return refined_output

# =====================================================================
# 4. DENSENET121 + CBAM MODEL
# =====================================================================
def build_densenet_cbam(num_classes=3, input_shape=(256, 256, 3), weights='imagenet'):
    """
    Builds the DenseNet121 + CBAM attention model.
    This configuration provides strong feature extraction and attention capability.
    """
    base_model = DenseNet121(
        include_top=False, weights=weights, input_shape=input_shape
    )
    base_model.trainable = False  # Freeze backbone initially

    x = base_model.output
    x = cbam_block(x)  # Apply CBAM attention
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

    print("✅ DenseNet121 + CBAM model built successfully.")
    return model

# =====================================================================
# 5. FINE-TUNING UTILITY (LOGGED)
# =====================================================================
def fine_tune_model(model, num_layers_to_unfreeze=50):
    """
    Enables fine-tuning for the top 'num_layers_to_unfreeze' layers of the base model.
    Automatically logs fine-tuning configuration to results/fine_tuning_log.txt
    """
    start = time.time()

    # Identify base model (DenseNet)
    base_model = None
    for layer in model.layers:
        if "densenet" in layer.name.lower():
            base_model = layer
            break
    if base_model is None:
        base_model = model

    # Unfreeze last N layers
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    # Recompile with a lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    elapsed = time.time() - start
    device = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"

    # Log fine-tuning configuration
    log_text = (
        f"Fine-tuning enabled\n"
        f"Unfrozen layers: {num_layers_to_unfreeze}\n"
        f"Trainable layers: {sum(l.trainable for l in base_model.layers)}\n"
        f"Device: {device}\n"
        f"Time: {elapsed:.2f}s\n"
    )
    log_path = os.path.join(RESULTS_DIR, "fine_tuning_log.txt")
    with open(log_path, "a") as f:
        f.write(log_text + "\n")

    print("🔓 Fine-tuning configured and logged.")
    return model

# =====================================================================
# 6. MODEL FACTORY ENTRY POINT
# =====================================================================
def get_model(
    model_name="DenseNet121",
    num_classes=3,
    input_shape=(256, 256, 3),
    weights="imagenet",
    export_summary=True
):
    """
    Builds and returns the DenseNet121 + CBAM model.
    Compatible with trainer.py calls that pass model_name.
    Automatically saves model architecture summary to results/model_summary.txt
    """
    if model_name != "DenseNet121":
        print(f"⚠️ Warning: '{model_name}' not supported. Using DenseNet121 + CBAM instead.")

    model = build_densenet_cbam(
        num_classes=num_classes,
        input_shape=input_shape,
        weights=weights
    )

    if export_summary:
        summary_path = os.path.join(RESULTS_DIR, "model_summary.txt")
        with open(summary_path, "w") as f:
            model.summary(print_fn=lambda x: f.write(x + "\n"))
        print(f"🧾 Model summary saved to {summary_path}")

    return model
