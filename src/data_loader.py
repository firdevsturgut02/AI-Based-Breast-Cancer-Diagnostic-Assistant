"""
data_loader.py
---------------
High-performance data loading and augmentation module for breast ultrasound classification.
Optimized for TÜBİTAK 2209-A Academic Projects.

Features:
- Fixed Albumentations v1.4+ compatibility (ElasticTransform & ShiftScaleRotate)
- Robust Class Balancing (Oversampling minority classes in training set)
- CLAHE & Medical-grade noise reduction for ultrasound images
- Stratified data splits for statistical validity
"""

# =====================================================================
# 1. IMPORTS
# =====================================================================
import os
import cv2
import numpy as np
import pandas as pd
import albumentations as A
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import matplotlib.pyplot as plt

# =====================================================================
# 2. CUSTOM DATA GENERATOR
# =====================================================================
class MedicalDataGenerator(Sequence):
    """
    Custom Keras data generator for breast ultrasound images.
    Supports real-time augmentation, normalization, and batching.
    """

    def __init__(self, df: pd.DataFrame, batch_size: int = 16,
                 img_size=(256, 256), augment: bool = False, shuffle: bool = True):
        self.df = df.copy()
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.shuffle = shuffle
        self.class_map = {label: i for i, label in enumerate(sorted(self.df['Label'].unique()))}
        self.n_classes = len(self.class_map)

        # Define augmentation pipeline (Fixed for Albumentations latest versions)
        if augment:
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
                # Fixed: ShiftScaleRotate uses newer interpolation standards
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, p=0.4),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
                # Fixed: Removed deprecated 'alpha_affine' to stop UserWarnings
                A.ElasticTransform(alpha=1, sigma=50, p=0.25),
                # Fixed: Updated GaussNoise to use standard range parameters
                A.GaussNoise(std_range=(0.01, 0.05), p=0.25),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225))
            ])
        else:
            self.aug = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225))
            ])

        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

        print(f"📦 Initialized generator: {len(self.df)} images, {self.n_classes} classes.")
        print(f"🧩 Augmentation: {'ON' if self.augment else 'OFF'} | Shuffle: {self.shuffle}")

    def __len__(self):
        """Number of batches per epoch."""
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        """Shuffle data after each epoch."""
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, idx):
        """Generate one batch of data."""
        batch_df = self.df.iloc[idx * self.batch_size: (idx + 1) * self.batch_size]
        X, y = [], []

        for _, row in batch_df.iterrows():
            img = cv2.imread(row['Path'])
            if img is None:
                continue  # Skip missing or unreadable files
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)

            # Apply augmentation
            img = self.aug(image=img)['image'].astype(np.float32)

            # One-hot encode labels
            label_idx = self.class_map[row['Label']]
            y_onehot = tf.keras.utils.to_categorical(label_idx, num_classes=self.n_classes)

            X.append(img)
            y.append(y_onehot)

        X = np.stack(X)
        y = np.stack(y)
        return X, y

# =====================================================================
# 3. DATA PREPARATION FUNCTIONS
# =====================================================================
def prepare_data_frames(data_dir: str, seed: int = 123, balance: bool = True):
    """
    Prepares train, validation, and test DataFrames from dataset directory.
    """

    data_paths, labels = [], []
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory {data_dir} not found!")

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                # Medical standard: Train on original images, ignore masks
                if file.lower().endswith(('.png', '.jpg', '.jpeg')) and 'mask' not in file.lower():
                    data_paths.append(os.path.join(folder_path, file))
                    labels.append(folder)

    df = pd.DataFrame({"Path": data_paths, "Label": labels})

    # Stratified splits for reproducibility and statistical validity
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["Label"], random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["Label"], random_state=seed)

    # Note: Only balance the training set to prevent evaluation bias
    if balance:
        train_df = balance_dataframe(train_df, seed)

    print(f"✅ Data prepared: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    describe_dataset(train_df, "Training Set (Final)")
    
    return train_df, val_df, test_df

# =====================================================================
# 4. CLASS BALANCING (Oversampling Majority Match)
# =====================================================================
def balance_dataframe(df: pd.DataFrame, seed: int = 42):
    """Balances dataset by oversampling minority classes to match the majority class count."""
    counts = df['Label'].value_counts()
    max_count = counts.max()
    
    balanced_list = []
    for label in counts.index:
        df_class = df[df['Label'] == label]
        df_class_resampled = resample(df_class, 
                                     replace=True, 
                                     n_samples=max_count, 
                                     random_state=seed)
        balanced_list.append(df_class_resampled)
    
    balanced_df = pd.concat(balanced_list)
    print(f"⚖️ Balanced all classes to {max_count} samples each. Total samples: {len(balanced_df)}")
    return balanced_df.sample(frac=1).reset_index(drop=True)

# =====================================================================
# 5. DATASET STATISTICS
# =====================================================================
def describe_dataset(df: pd.DataFrame, name: str = "Dataset"):
    """Displays class distribution percentages for reporting/logging."""
    summary = (df['Label'].value_counts(normalize=True) * 100).round(2)
    print(f"\n📊 {name} distribution (%):")
    print(summary.to_string())

# =====================================================================
# 6. AUGMENTATION VISUALIZATION
# =====================================================================
def visualize_augmentations(generator, n=5):
    """
    Displays sample augmentations from the generator.
    
    """
    X, y = generator.__getitem__(0)
    plt.figure(figsize=(15, 3))
    for i in range(min(n, len(X))):
        plt.subplot(1, n, i + 1)
        # Denormalize for visualization purposes
        img_vis = X[i].copy()
        img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-5)
        plt.imshow(img_vis)
        label = list(generator.class_map.keys())[np.argmax(y[i])]
        plt.title(f"Class: {label}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# =====================================================================
# 7. MAIN TEST
# =====================================================================
if __name__ == "__main__":
    data_dir = "Dataset_BUSI_with_GT"
    train_df, val_df, test_df = prepare_data_frames(data_dir, balance=True)

    # Initialize a generator and visualize
    gen = MedicalDataGenerator(train_df, augment=True)
    visualize_augmentations(gen, n=5)
