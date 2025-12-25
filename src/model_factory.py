"""
model_factory.py
----------------
Hybrid CNN Architecture for Breast Ultrasound Classification.
Combines DenseNet121 with the Convolutional Block Attention Module (CBAM).

Technical Highlights:
    ✅ Cross-backend compatibility using keras.ops
    ✅ Integrated Channel & Spatial Attention (CBAM)
    ✅ Automated model visualization and structured summary export
    ✅ End-to-end trainable DenseNet backbone for 50-epoch optimization

Author: [Your Name]
Affiliation: [Your Institution / Research Group]
Date: [Auto-generated]
"""

# =====================================================================
# 1. IMPORTS
# =====================================================================
import os
import time
import tensorflow as tf
from keras import ops
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization,
    Conv2D, Multiply, Add, Activation, Reshape,
    GlobalMaxPooling2D, Input
)
from tensorflow.keras.models import Model


# =====================================================================
# 2. OUTPUT DIRECTORY CONFIGURATION
# =====================================================================
BASE_RESULTS = "results"
MODEL_FACTORY_DIR = os.path.join(BASE_RESULTS, "model_factory")
os.makedirs(MODEL_FACTORY_DIR, exist_ok=True)


# =====================================================================
# 3. CBAM MODULE
# =====================================================================
def cbam_block(input_tensor, ratio: int = 8):
    """
    Implements the Convolutional Block Attention Module (CBAM).
    Enhances representational power via sequential Channel & Spatial Attention.

    Args:
        input_tensor: Feature map input tensor.
        ratio: Reduction ratio for channel attention.

    Returns:
        Refined feature map after channel and spatial attention operations.
    """
    channel = input_tensor.shape[-1]

    # ---- Channel Attention ----
    shared_dense_1 = Dense(channel // ratio, activation='relu',
                           kernel_initializer='he_normal', use_bias=False)
    shared_dense_2 = Dense(channel, kernel_initializer='he_normal', use_bias=False)

    # Global Average Pooling
    avg_pool = GlobalAveragePooling2D()(input_tensor)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_dense_2(shared_dense_1(avg_pool))

    # Global Max Pooling
    max_pool = GlobalMaxPooling2D()(input_tensor)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_dense_2(shared_dense_1(max_pool))

    # Combine and activate
    channel_attention = Add()([avg_pool, max_pool])
    channel_attention = Activation('sigmoid')(channel_attention)
    channel_refined = Multiply()([input_tensor, channel_attention])

    # ---- Spatial Attention ----
    avg_spatial = ops.mean(channel_refined, axis=-1, keepdims=True)
    max_spatial = ops.amax(channel_refined, axis=-1, keepdims=True)
    concat = ops.concatenate([avg_spatial, max_spatial], axis=-1)

    spatial_attention = Conv2D(1, kernel_size=7, strides=1, padding='same',
                               activation='sigmoid', kernel_initializer='he_normal')(concat)
    refined_output = Multiply()([channel_refined, spatial_attention])

    return refined_output


# =====================================================================
# 4. MODEL CONSTRUCTION (DenseNet121 + CBAM)
# =====================================================================
def build_densenet_cbam(
    num_classes: int = 3,
    input_shape=(256, 256, 3),
    weights: str = "imagenet"
) -> Model:
    """
    Constructs a hybrid DenseNet121-CBAM model for medical image classification.

    Args:
        num_classes: Number of output diagnostic categories.
        input_shape: Expected input tensor shape.
        weights: Pretraining source for DenseNet backbone ('imagenet' or None).

    Returns:
        A compiled tf.keras Model instance.
    """
    inputs = Input(shape=input_shape)
    base_model = DenseNet121(include_top=False, weights=weights, input_tensor=inputs)

    # Enable fine-tuning of all layers
    base_model.trainable = True

    # Apply CBAM after final DenseNet convolutional block
    x = cbam_block(base_model.output)

    # Classification head
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu', kernel_initializer='he_normal')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name="DenseNet121_CBAM")

    # Optimizer and metrics setup
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    print("✅ DenseNet121 + CBAM hybrid model successfully constructed.")
    return model


# =====================================================================
# 5. FACTORY WRAPPER & DOCUMENTATION EXPORT
# =====================================================================
def get_model(
    num_classes: int = 3,
    input_shape=(256, 256, 3),
    export_outputs: bool = True
) -> Model:
    """
    Factory wrapper to construct, compile, and document the model.

    Automatically saves:
        - Model summary (TXT)
        - Architectural diagram (PNG)
    """
    model = build_densenet_cbam(num_classes=num_classes, input_shape=input_shape)

    if export_outputs:
        # Export model summary
        summary_path = os.path.join(MODEL_FACTORY_DIR, "model_summary.txt")
        with open(summary_path, "w") as f:
            model.summary(print_fn=lambda x: f.write(x + "\n"))

        # Export architecture diagram
        try:
            plot_path = os.path.join(MODEL_FACTORY_DIR, "architecture_diagram.png")
            tf.keras.utils.plot_model(
                model, to_file=plot_path,
                show_shapes=True, show_layer_names=True,
                dpi=120
            )
            print(f"🖼️ Architectural diagram exported → {plot_path}")
        except Exception as e:
            print(f"⚠️ Diagram export failed (missing Graphviz/Pydot): {e}")

        print(f"🧾 Model summary saved at → {summary_path}")

    return model


# =====================================================================
# 6. INTERNAL VALIDATION ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    start_time = time.time()
    model = get_model()
    print(f"⏱️ Model initialization completed in {time.time() - start_time:.2f}s")
