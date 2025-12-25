"""
data_loader.py
-------------------------------------------------
Advanced Data Loading, Statistical Analysis, and Augmentation Module
for Breast Ultrasound Classification (Technical Paper Grade).

Outputs are stored in: results/data_loader/
Features:
✅ Stratified Splitting (Train/Val/Test)
✅ Scientific Class Balancing (Oversampling)
✅ High-Resolution Statistical Plots
✅ Exemplary Augmentation Samples
-------------------------------------------------
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
from matplotlib.backends.backend_pdf import PdfPages

# =====================================================================
# 2. DIRECTORY CONFIGURATION
# =====================================================================
BASE_RESULTS = "results"
DATA_LOADER_DIR = os.path.join(BASE_RESULTS, "data_loader")
os.makedirs(DATA_LOADER_DIR, exist_ok=True)

# =====================================================================
# 3. CUSTOM DATA GENERATOR
# =====================================================================
class MedicalDataGenerator(Sequence):
    """
    Standardized Keras data generator for medical imaging.
    Implements real-time augmentation and ImageNet normalization.
    """
    def __init__(self, df: pd.DataFrame, batch_size: int = 16, 
                 img_size=(256, 256), augment: bool = False, shuffle: bool = True):
        self.df = df.copy()
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.shuffle = shuffle
        self.class_map = {label: i for i, label in enumerate(sorted(self.df["Label"].unique()))}
        self.n_classes = len(self.class_map)

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
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, idx):
        batch_df = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, y = [], []
        for _, row in batch_df.iterrows():
            img = cv2.imread(row["Path"])
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            img = self.aug(image=img)["image"].astype(np.float32)
            y_onehot = tf.keras.utils.to_categorical(self.class_map[row["Label"]], self.n_classes)
            X.append(img); y.append(y_onehot)
        return np.stack(X), np.stack(y)

# =====================================================================
# 4. VISUALIZATION ENGINE (TECHNICAL PAPER STANDARDS)
# =====================================================================
def plot_class_distribution(df: pd.DataFrame, stage: str):
    """Generates publication-quality bar charts for class balance analysis."""
    plt.figure(figsize=(10, 6))
    counts = df["Label"].value_counts().sort_index()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = plt.bar(counts.index, counts.values, color=colors, edgecolor='black', alpha=0.8)
    
    plt.title(f"Dataset Distribution: {stage.replace('_', ' ')}", fontsize=14, fontweight='bold')
    plt.xlabel("Pathological Class", fontsize=12)
    plt.ylabel("Number of Ultrasound Samples", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 5, yval, ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(DATA_LOADER_DIR, f"distribution_{stage.lower()}.png")
    plt.savefig(path, dpi=300); plt.close()

def plot_image_resolution_analysis(df: pd.DataFrame):
    """Analyzes the resolution distribution of the source imagery."""
    widths, heights = [], []
    for path in df["Path"].sample(min(150, len(df))):
        img = cv2.imread(path)
        if img is not None:
            h, w = img.shape[:2]; widths.append(w); heights.append(h)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(widths, heights, alpha=0.5, c='darkred', edgecolors='white', s=80)
    plt.title("Spatial Resolution Distribution of Input Images", fontsize=14, fontweight='bold')
    plt.xlabel("Width (Pixels)", fontsize=12); plt.ylabel("Height (Pixels)", fontsize=12)
    plt.axvline(np.mean(widths), color='blue', linestyle='--', label=f'Mean Width: {int(np.mean(widths))}')
    plt.axhline(np.mean(heights), color='green', linestyle='--', label=f'Mean Height: {int(np.mean(heights))}')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    
    path = os.path.join(DATA_LOADER_DIR, "resolution_analysis.png")
    plt.savefig(path, dpi=300); plt.close()

def visualize_samples(df: pd.DataFrame, n_samples=5):
    """Displays raw samples from the dataset for morphological check."""
    plt.figure(figsize=(18, 4))
    samples = df.sample(n_samples)
    for i, (_, row) in enumerate(samples.iterrows()):
        img = cv2.imread(row["Path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, n_samples, i+1)
        plt.imshow(img)
        plt.title(f"Class: {row['Label']}", fontsize=10, fontweight='bold')
        plt.axis("off")
    plt.tight_layout()
    path = os.path.join(DATA_LOADER_DIR, "dataset_samples.png")
    plt.savefig(path, dpi=300); plt.close()

def visualize_augmentation_impact(generator, n=5):
    """Demonstrates the effect of the data augmentation pipeline."""
    X, _ = generator.__getitem__(0)
    plt.figure(figsize=(18, 4))
    for i in range(min(n, len(X))):
        img = X[i]
        # De-normalize for visualization
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        plt.subplot(1, n, i + 1)
        plt.imshow(img)
        plt.title("Augmented Pipeline Output", fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    path = os.path.join(DATA_LOADER_DIR, "augmentation_samples.png")
    plt.savefig(path, dpi=300); plt.close()

# =====================================================================
# 5. CORE PREPARATION LOGIC
# =====================================================================
def prepare_data_frames(data_dir: str, seed: int = 42, balance: bool = True):
    paths, labels = [], []
    for cls in os.listdir(data_dir):
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path): continue
        for file in os.listdir(cls_path):
            if file.lower().endswith((".png", ".jpg", ".jpeg")) and "mask" not in file.lower():
                paths.append(os.path.join(cls_path, file)); labels.append(cls)
    
    df = pd.DataFrame({"Path": paths, "Label": labels})
    
    # Stratified Split (80% Train, 10% Val, 10% Test)
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["Label"], random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["Label"], random_state=seed)

    plot_class_distribution(train_df, "Initial_Training_Set")
    
    if balance:
        counts = train_df["Label"].value_counts()
        max_samples = counts.max()
        balanced = []
        for label in counts.index:
            df_cls = train_df[train_df["Label"] == label]
            balanced.append(resample(df_cls, replace=True, n_samples=max_samples, random_state=seed))
        train_df = pd.concat(balanced).sample(frac=1).reset_index(drop=True)
        plot_class_distribution(train_df, "Balanced_Training_Set")

    return train_df, val_df, test_df

# =====================================================================
# 6. MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":
    DATA_PATH = "Dataset_BUSI_with_GT"
    
    print("📈 Analyzing Ultrasound Dataset...")
    train_df, val_df, test_df = prepare_data_frames(DATA_PATH)
    
    print("🖼️ Generating Technical Visualizations...")
    plot_image_resolution_analysis(train_df)
    visualize_samples(train_df)
    
    gen = MedicalDataGenerator(train_df, augment=True)
    visualize_augmentation_impact(gen)
    
    print(f"✅ Data Preparation Success. Reports saved in: {DATA_LOADER_DIR}/")
