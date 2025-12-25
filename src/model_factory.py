"""
model_factory.py
----------------
Model architecture construction module for breast cancer classification.
Implements DenseNet121 integrated with Convolutional Block Attention Module (CBAM).

Technical Features:
✅ Keras 3 Native Operations (keras.ops) for cross-backend compatibility.
✅ Integrated Channel and Spatial Attention mechanisms.
✅ Automated architectural visualization and summary logging.
✅ Optimized for single-stage 50-epoch full-network training.
"""

# =====================================================================
# 1. IMPORTS
# =====================================================================
import os
import time
import tensorflow as tf
from keras import ops  # Keras 3 operations for tensor manipulation
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization,
    Conv2D, Multiply, Add, Activation, Reshape, GlobalMaxPooling2D, Input
)
from tensorflow.keras.models import Model

# =====================================================================
# 2. DIRECTORY CONFIGURATION
# =====================================================================
BASE_RESULTS = "results"
MODEL_FACTORY_DIR = os.path.join(BASE_RESULTS, "model_factory")
os.makedirs(MODEL_FACTORY_DIR, exist_ok=True)

# =====================================================================
# 3. CBAM ATTENTION MECHANISM
# =====================================================================
def cbam_block(input_tensor, ratio=8):
    """
    Implementation of Convolutional Block Attention Module (CBAM).
    Refines feature maps through both channel and spatial dimensions.
    """
    channel = input_tensor.shape[-1]

    # --- Channel Attention Sub-module ---
    # Shared MLP layers
    shared_dense_1 = Dense(channel // ratio, activation='relu', use_bias=False, kernel_initializer='he_normal')
    shared_dense_2 = Dense(channel, use_bias=False, kernel_initializer='he_normal')

    # Avg and Max pooling paths
    avg_pool = GlobalAveragePooling2D()(input_tensor)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_dense_2(shared_dense_1(avg_pool))

    max_pool = GlobalMaxPooling2D()(input_tensor)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_dense_2(shared_dense_1(max_pool))

    # Element-wise summation and sigmoid activation
    channel_attention = Add()([avg_pool, max_pool])
    channel_attention = Activation('sigmoid')(channel_attention)
    channel_refined = Multiply()([input_tensor, channel_attention])

    # --- Spatial Attention Sub-module ---
    # Global statistics across the channel dimension using Keras ops
    avg_spatial = ops.mean(channel_refined, axis=-1, keepdims=True)
    max_spatial = ops.amax(channel_refined, axis=-1, keepdims=True)
    
    # Concatenation and spatial convolution
    concat = ops.concatenate([avg_spatial, max_spatial], axis=-1)
    spatial_attention = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid', kernel_initializer='he_normal')(concat)

    # Final refined output
    refined_output = Multiply()([channel_refined, spatial_attention])
    return refined_output

# =====================================================================
# 4. MODEL CONSTRUCTION (DENSENET121 + CBAM)
# =====================================================================
def build_densenet_cbam(num_classes=3, input_shape=(256, 256, 3), weights='imagenet'):
    """
    Assembles the hybrid DenseNet121-CBAM model for pathological classification.
    """
    inputs = Input(shape=input_shape)
    
    # Pre-trained DenseNet121 backbone
    base_model = DenseNet121(include_top=False, weights=weights, input_tensor=inputs)
    
    # Enable full model training for the requested 50-epoch run
    base_model.trainable = True 

    # Integration of CBAM at the end of the feature extraction layers
    x = base_model.output
    x = cbam_block(x) 
    
    # Classification Head
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu', kernel_initializer='he_normal')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    
    # Compiler Configuration
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    print("✅ DenseNet121 + CBAM architecture constructed successfully.")
    return model

# =====================================================================
# 5. ENTRY POINT & VISUALIZATION
# =====================================================================
def get_model(num_classes=3, input_shape=(256, 256, 3), export_outputs=True):
    """
    Factory method to retrieve the model and generate technical documentations.
    """
    model = build_densenet_cbam(num_classes=num_classes, input_shape=input_shape)

    if export_outputs:
        # Save structural summary
        summary_path = os.path.join(MODEL_FACTORY_DIR, "model_summary.txt")
        with open(summary_path, "w") as f:
            model.summary(print_fn=lambda x: f.write(x + "\n"))
        
        # Save architectural diagram (Graphviz/Pydot required)
        try:
            plot_path = os.path.join(MODEL_FACTORY_DIR, "architecture_diagram.png")
            tf.keras.utils.plot_model(model, to_file=plot_path, show_shapes=True, show_layer_names=True)
            print(f"🖼️ Architecture diagram exported to {plot_path}")
        except Exception as e:
            print(f"⚠️ Diagram export failed (Check pydot/graphviz): {e}")

        print(f"🧾 Model summary saved to {summary_path}")

    return model

if __name__ == "__main__":
    # Internal validation
    get_model()
