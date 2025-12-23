"""
model_factory.py
----------------
Model builder module for breast ultrasound classification.

Features:
- DenseNet121 (baseline)
- EfficientNetB3 (for comparison)
- DenseNet121 + CBAM attention (for explainability and performance)
- Fine-tuning utilities
- Architecture summary export
"""

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
# 1. CBAM ATTENTION BLOCK
# =====================================================================
def cbam_block(input_tensor, ratio=8):
    """Convolutional Block Attention Module (CBAM)."""
    channel = input_tensor.shape[-1]
    shared_dense_one = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)
    shared_dense_two = Dense(channel, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)

    # Channel Attention
    avg_pool = GlobalAveragePooling2D()(input_tensor)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_dense_two(shared_dense_one(avg_pool))

    max_pool = GlobalMaxPooling2D()(input_tensor)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_dense_two(shared_dense_one(max_pool))

    channel_attention = Add()([avg_pool, max_pool])
    channel_attention = Activation('sigmoid')(channel_attention)
    channel_refined = Multiply()([input_tensor, channel_attention])

    # Spatial Attention
    avg_pool = tf.reduce_mean(channel_refined, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(channel_refined, axis=-1, keepdims=True)
    concat = tf.concat([avg_pool, max_pool], axis=-1)
    spatial_attention = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)
    refined_output = Multiply()([channel_refined, spatial_attention])

    return refined_output


# =====================================================================
# 2. DENSENET121 BUILDER
# =====================================================================
def build_densenet121(num_classes=3, input_shape=(256, 256, 3), weights='imagenet'):
    """Builds DenseNet121 baseline model."""
    base_model = DenseNet121(include_top=False, weights=weights, input_shape=input_shape)
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    print("✅ DenseNet121 baseline model built successfully.")
    return model


# =====================================================================
# 3. EFFICIENTNETB3 BUILDER
# =====================================================================
def build_efficientnetb3(num_classes=3, input_shape=(256, 256, 3), weights='imagenet'):
    """Builds EfficientNetB3 model for performance comparison."""
    base_model = EfficientNetB3(include_top=False, weights=weights, input_shape=input_shape)
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    print("✅ EfficientNetB3 model built successfully.")
    return model


# =====================================================================
# 4. DENSENET121 + CBAM BUILDER
# =====================================================================
def build_densenet_cbam(num_classes=3, input_shape=(256, 256, 3), weights='imagenet'):
    """Builds DenseNet121 model enhanced with CBAM attention."""
    base_model = DenseNet121(include_top=False, weights=weights, input_shape=input_shape)
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

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    print("✅ DenseNet121 + CBAM model built successfully.")
    return model


# =====================================================================
# 5. FINE-TUNING UTILITY
# =====================================================================
def fine_tune_model(model, num_layers_to_unfreeze=50):
    """Safely unfreezes the last N layers for fine-tuning."""
    start = time.time()

    # Try to locate base CNN
    base_model = None
    for layer in model.layers:
        if any(name in layer.name.lower() for name in ['densenet', 'efficientnet']):
            base_model = layer
            break
    if base_model is None:
        print("⚠️ Base model not found; unfreezing entire model.")
        base_model = model

    base_model.trainable = True
    total_layers = len(base_model.layers)
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)

    # Recompile with lower LR
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    elapsed = time.time() - start
    device = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
    print(f"\n🔓 Fine-tuning {num_layers_to_unfreeze} layers | Total: {total_layers} | Trainable: {trainable_count} | Device: {device} | Time: {elapsed:.2f}s")
    return model


# =====================================================================
# 6. MODEL FACTORY ENTRY POINT
# =====================================================================
def get_model(model_name="densenet121", num_classes=3, input_shape=(256, 256, 3), weights='imagenet', export_summary=True):
    """
    Factory function to get a specific architecture by name.
    Options: ['densenet121', 'efficientnetb3', 'densenet_cbam']
    """
    model_map = {
        "densenet121": build_densenet121,
        "efficientnetb3": build_efficientnetb3,
        "densenet_cbam": build_densenet_cbam,
    }

    if model_name.lower() not in model_map:
        raise ValueError(f"❌ Unknown model name '{model_name}'. Choose from {list(model_map.keys())}")

    model = model_map[model_name.lower()](num_classes=num_classes, input_shape=input_shape, weights=weights)

    # Optionally export summary for paper
    if export_summary:
        summary_path = os.path.join("models", f"{model_name}_summary.txt")
        os.makedirs("models", exist_ok=True)
        with open(summary_path, "w") as f:
            model.summary(print_fn=lambda x: f.write(x + "\n"))
        print(f"🧾 Model summary saved to {summary_path}")

    return model
