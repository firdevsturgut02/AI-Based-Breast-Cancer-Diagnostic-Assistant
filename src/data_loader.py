"""
data_loader.py
---------------
High-performance data loading and augmentation module for breast ultrasound classification.


Author: <Your Name>
Email: <Your Email>
License: CC BY-NC 4.0

Features:
- CLAHE, elastic deformation, and Gaussian noise augmentations
- Stratified data splits for reproducibility
- Custom Keras Sequence generator
- Real-time augmentation via Albumentations
- Optional class balancing
- Visualization utilities for article figures
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

        # Define augmentation pipeline (Albumentations)
        if augment:
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomBrightnessContrast(p=0.4),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=20, p=0.4),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.25),
                A.GaussNoise(var_limit=(10, 50), p=0.25),
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
def prepare_data_frames(data_dir: str, seed: int = 123, balance: bool = False):
    """
    Prepares train, validation, and test DataFrames from dataset directory.
    Expected structure:
        Dataset_BUSI_with_GT/
            benign/
            malignant/
            normal/
    """

    data_paths, labels = [], []
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')) and 'mask' not in file.lower():
                    data_paths.append(os.path.join(folder_path, file))
                    labels.append(folder)

    df = pd.DataFrame({"Path": data_paths, "Label": labels})

    # Stratified splits for reproducibility
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["Label"], random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["Label"], random_state=seed)

    if balance:
        train_df = balance_dataframe(train_df)

    print(f"✅ Data prepared: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    describe_dataset(df)

    return train_df, val_df, test_df

# =====================================================================
# 4. CLASS BALANCING (OPTIONAL)
# =====================================================================
def balance_dataframe(df: pd.DataFrame):
    """Balances dataset by oversampling minority classes."""
    min_class = df['Label'].value_counts().max()
    balanced_df = pd.concat([
        resample(df[df['Label'] == c], replace=True, n_samples=min_class, random_state=42)
        for c in df['Label'].unique()
    ])
    print(f"⚖️ Balanced dataset to {len(balanced_df)} total samples.")
    return balanced_df.sample(frac=1).reset_index(drop=True)

# =====================================================================
# 5. DATASET STATISTICS
# =====================================================================
def describe_dataset(df: pd.DataFrame):
    """Displays class distribution percentages."""
    summary = (df['Label'].value_counts(normalize=True) * 100).round(2)
    print("\n📊 Class distribution (%):")
    print(summary.to_string())

# =====================================================================
# 6. AUGMENTATION VISUALIZATION (for article/presentation)
# =====================================================================
def visualize_augmentations(generator, n=5):
    """
    Displays sample augmentations from the generator.
    Use this for figures in your paper or presentation.
    """
    X, y = generator.__getitem__(0)
    plt.figure(figsize=(15, 3))
    for i in range(min(n, len(X))):
        plt.subplot(1, n, i + 1)
        img_vis = (X[i] - X[i].min()) / (X[i].max() - X[i].min())
        plt.imshow(img_vis)
        plt.title(list(generator.class_map.keys())[np.argmax(y[i])])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# =====================================================================
# 7. MAIN TEST (OPTIONAL)
# =====================================================================
if __name__ == "__main__":
    data_dir = "Dataset_BUSI_with_GT"
    train_df, val_df, test_df = prepare_data_frames(data_dir, balance=True)

    # Initialize a generator and visualize
    gen = MedicalDataGenerator(train_df, augment=True)
    visualize_augmentations(gen, n=5)
