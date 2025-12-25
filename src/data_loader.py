"""
data_loader.py
-------------------------------------------------
Advanced Data Loading, Statistical Analysis, and Augmentation Module
for Breast Ultrasound Classification (Academic Research Standard).

Version: 2.0
Outputs Directory: results/data_loader/

Core Features:
    ✅ Stratified Data Splitting (Train/Val/Test)
    ✅ Class Balancing via Statistical Oversampling
    ✅ Scientific Visualization (Class Distribution, Resolution Analysis)
    ✅ Controlled Augmentation Pipeline with ImageNet Normalization
    ✅ Reproducible and Logging-Based Workflow
-------------------------------------------------
"""

# =====================================================================
# 1. IMPORTS
# =====================================================================
import os
import cv2
import logging
import numpy as np
import pandas as pd
import albumentations as A
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from typing import Tuple, List
from datetime import datetime

# =====================================================================
# 2. GLOBAL CONFIGURATION & LOGGING
# =====================================================================
BASE_RESULTS = "results"
DATA_LOADER_DIR = os.path.join(BASE_RESULTS, "data_loader")
os.makedirs(DATA_LOADER_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(DATA_LOADER_DIR, "data_loader_log.txt"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

logging.info("=== Data Loader Initialized (Academic Grade) ===")


# =====================================================================
# 3. CUSTOM DATA GENERATOR
# =====================================================================
class MedicalDataGenerator(Sequence):
    """
    Standardized TensorFlow Sequence for Medical Image Datasets.

    Attributes:
        df (pd.DataFrame): Image paths and labels.
        batch_size (int): Number of samples per batch.
        img_size (tuple): Target image dimensions (H, W).
        augment (bool): Whether to apply augmentation.
        shuffle (bool): Shuffle data after each epoch.
        class_map (dict): Label-to-index mapping.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int = 16,
        img_size: Tuple[int, int] = (256, 256),
        augment: bool = False,
        shuffle: bool = True,
    ):
        self.df = df.copy()
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.shuffle = shuffle
        self.class_map = {label: i for i, label in enumerate(sorted(self.df["Label"].unique()))}
        self.n_classes = len(self.class_map)

        # Define augmentation strategy
        if self.augment:
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, p=0.4),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.25),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        else:
            self.aug = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

        if self.shuffle:
            self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)

    def __len__(self) -> int:
        """Total number of batches per epoch."""
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        """Shuffle dataset after each epoch for improved generalization."""
        if self.shuffle:
            self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate one batch of data."""
        batch_df = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, y = [], []
        for _, row in batch_df.iterrows():
            img = cv2.imread(row["Path"])
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            img = self.aug(image=img)["image"].astype(np.float32)
            y_onehot = tf.keras.utils.to_categorical(self.class_map[row["Label"]], self.n_classes)
            X.append(img)
            y.append(y_onehot)
        return np.stack(X), np.stack(y)


# =====================================================================
# 4. VISUALIZATION MODULE (RESEARCH-GRADE PLOTS)
# =====================================================================
def plot_class_distribution(df: pd.DataFrame, stage: str):
    """Generate high-quality bar plot for class distribution."""
    plt.figure(figsize=(10, 6))
    counts = df["Label"].value_counts().sort_index()
    sns.barplot(x=counts.index, y=counts.values, palette="viridis", edgecolor="black")
    plt.title(f"Class Distribution: {stage}", fontsize=14, fontweight='bold')
    plt.xlabel("Pathological Class", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    for i, val in enumerate(counts.values):
        plt.text(i, val + 2, f"{val}", ha='center', fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(DATA_LOADER_DIR, f"distribution_{stage.lower().replace(' ', '_')}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    logging.info(f"📊 Saved distribution plot for {stage} → {out_path}")


def plot_image_resolution_analysis(df: pd.DataFrame):
    """Generate scatter plot of image resolutions to assess spatial uniformity."""
    widths, heights = [], []
    for path in df["Path"].sample(min(150, len(df))):
        img = cv2.imread(path)
        if img is not None:
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)

    plt.figure(figsize=(10, 6))
    plt.scatter(widths, heights, alpha=0.6, color="#e74c3c", edgecolors="white", s=80)
    plt.title("Resolution Distribution of Source Images", fontsize=14, fontweight="bold")
    plt.xlabel("Width (px)")
    plt.ylabel("Height (px)")
    plt.axvline(np.mean(widths), color='blue', linestyle='--', label=f"Mean Width = {int(np.mean(widths))}")
    plt.axhline(np.mean(heights), color='green', linestyle='--', label=f"Mean Height = {int(np.mean(heights))}")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(DATA_LOADER_DIR, "resolution_analysis.png")
    plt.savefig(path, dpi=300)
    plt.close()
    logging.info(f"📐 Saved resolution analysis → {path}")


def visualize_samples(df: pd.DataFrame, n_samples: int = 5):
    """Display representative dataset samples (for morphology validation)."""
    plt.figure(figsize=(18, 4))
    samples = df.sample(n_samples)
    for i, (_, row) in enumerate(samples.iterrows()):
        img = cv2.imread(row["Path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, n_samples, i + 1)
        plt.imshow(img)
        plt.title(f"Class: {row['Label']}", fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    path = os.path.join(DATA_LOADER_DIR, "dataset_samples.png")
    plt.savefig(path, dpi=300)
    plt.close()
    logging.info(f"🧠 Sample visualization saved → {path}")


def visualize_augmentation_impact(generator: MedicalDataGenerator, n: int = 5):
    """Visualize augmentation results for verification of variability."""
    X, _ = generator.__getitem__(0)
    plt.figure(figsize=(18, 4))
    for i in range(min(n, len(X))):
        img = X[i]
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        plt.subplot(1, n, i + 1)
        plt.imshow(img)
        plt.title("Augmented Sample", fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    path = os.path.join(DATA_LOADER_DIR, "augmentation_samples.png")
    plt.savefig(path, dpi=300)
    plt.close()
    logging.info(f"🧩 Augmentation visualization saved → {path}")


# =====================================================================
# 5. DATA PREPARATION PIPELINE
# =====================================================================
def prepare_data_frames(
    data_dir: str,
    seed: int = 42,
    balance: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load dataset, apply stratified splitting and class balancing.

    Args:
        data_dir (str): Path to dataset root folder.
        seed (int): Random seed for reproducibility.
        balance (bool): Apply oversampling for class balance.

    Returns:
        (train_df, val_df, test_df)
    """
    paths, labels = [], []
    for cls in os.listdir(data_dir):
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for file in os.listdir(cls_path):
            if file.lower().endswith((".png", ".jpg", ".jpeg")) and "mask" not in file.lower():
                paths.append(os.path.join(cls_path, file))
                labels.append(cls)

    df = pd.DataFrame({"Path": paths, "Label": labels})
    logging.info(f"📁 Total samples loaded: {len(df)} | Classes: {df['Label'].nunique()}")

    # Stratified split (80/10/10)
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["Label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["Label"], random_state=seed
    )

    plot_class_distribution(train_df, "Initial Training Set")

    # Balance training data if required
    if balance:
        logging.info("⚖️ Applying class balancing (oversampling)...")
        counts = train_df["Label"].value_counts()
        max_samples = counts.max()
        balanced_frames = []
        for label in counts.index:
            cls_df = train_df[train_df["Label"] == label]
            balanced = resample(cls_df, replace=True, n_samples=max_samples, random_state=seed)
            balanced_frames.append(balanced)
        train_df = pd.concat(balanced_frames).sample(frac=1, random_state=seed).reset_index(drop=True)
        plot_class_distribution(train_df, "Balanced Training Set")

    logging.info(f"📊 Final split → Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


# =====================================================================
# 6. MAIN EXECUTION (STANDALONE MODE)
# =====================================================================
if __name__ == "__main__":
    DATA_PATH = "Dataset_BUSI_with_GT"
    logging.info("🔍 Performing exploratory data analysis on BUSI dataset...")

    train_df, val_df, test_df = prepare_data_frames(DATA_PATH)

    plot_image_resolution_analysis(train_df)
    visualize_samples(train_df)

    gen = MedicalDataGenerator(train_df, augment=True)
    visualize_augmentation_impact(gen)

    logging.info(f"✅ Data preparation and visualization complete. Results → {DATA_LOADER_DIR}")
