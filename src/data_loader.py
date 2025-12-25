"""
data_loader.py
-------------------------------------------------
High-performance data loading and augmentation module
for breast ultrasound image classification.

Features:
✅ Class distribution visualization (before & after balancing)
✅ Image resolution and augmentation checks
✅ All plots saved in results/
✅ Auto-generated PDF report summarizing dataset insights
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
# 2. GLOBAL PATHS
# =====================================================================
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# =====================================================================
# 3. CUSTOM DATA GENERATOR
# =====================================================================
class MedicalDataGenerator(Sequence):
    """
    Custom Keras data generator for breast ultrasound images.
    Supports real-time augmentation, normalization, and batching.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int = 16,
        img_size=(256, 256),
        augment: bool = False,
        shuffle: bool = True,
    ):
        self.df = df.copy()
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.shuffle = shuffle

        self.class_map = {
            label: i for i, label in enumerate(sorted(self.df["Label"].unique()))
        }
        self.n_classes = len(self.class_map)

        # Augmentation pipeline
        if self.augment:
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomBrightnessContrast(0.2, 0.2, p=0.4),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, p=0.4),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, p=0.25),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.25),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        else:
            self.aug = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

        print(f"[INFO] Generator initialized | Samples: {len(self.df)} | Classes: {self.n_classes}")
        print(f"[INFO] Augmentation: {'ON' if self.augment else 'OFF'}")

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
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            img = self.aug(image=img)["image"].astype(np.float32)

            label_idx = self.class_map[row["Label"]]
            y_onehot = tf.keras.utils.to_categorical(label_idx, self.n_classes)

            X.append(img)
            y.append(y_onehot)

        return np.stack(X), np.stack(y)

# =====================================================================
# 4. DATA PREPARATION
# =====================================================================
def prepare_data_frames(data_dir: str, seed: int = 123, balance: bool = True):
    """
    Creates stratified train/validation/test DataFrames and
    automatically saves visual reports before & after balancing.
    """

    paths, labels = [], []

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    for cls in os.listdir(data_dir):
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        for file in os.listdir(cls_path):
            if file.lower().endswith((".png", ".jpg", ".jpeg")) and "mask" not in file.lower():
                paths.append(os.path.join(cls_path, file))
                labels.append(cls)

    df = pd.DataFrame({"Path": paths, "Label": labels})

    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["Label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["Label"], random_state=seed
    )

    # BEFORE BALANCING
    plot_class_distribution(train_df, "Before_Balancing")

    if balance:
        train_df = balance_dataframe(train_df, seed)
        plot_class_distribution(train_df, "After_Balancing")

    print(f"[INFO] Dataset prepared | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df

# =====================================================================
# 5. CLASS BALANCING
# =====================================================================
def balance_dataframe(df: pd.DataFrame, seed: int = 42):
    """
    Oversamples minority classes to match majority class size.
    """
    counts = df["Label"].value_counts()
    max_count = counts.max()

    balanced = []
    for label in counts.index:
        df_cls = df[df["Label"] == label]
        df_resampled = resample(df_cls, replace=True, n_samples=max_count, random_state=seed)
        balanced.append(df_resampled)

    balanced_df = pd.concat(balanced).sample(frac=1).reset_index(drop=True)
    print(f"[INFO] Class balancing applied | Samples per class: {max_count}")
    return balanced_df

# =====================================================================
# 6. VISUALIZATION HELPERS
# =====================================================================
def plot_class_distribution(df: pd.DataFrame, name: str):
    plt.figure(figsize=(6, 4))
    df["Label"].value_counts().plot(kind="bar", color="teal")
    plt.title(f"Class Distribution – {name}")
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"class_distribution_{name.lower()}.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[SAVED] {path}")

def visualize_sample_images(df: pd.DataFrame, samples_per_class: int = 3):
    classes = df["Label"].unique()
    plt.figure(figsize=(samples_per_class * 3, len(classes) * 3))
    idx = 1
    for cls in classes:
        subset = df[df["Label"] == cls].sample(samples_per_class)
        for _, row in subset.iterrows():
            img = cv2.imread(row["Path"])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(len(classes), samples_per_class, idx)
            plt.imshow(img)
            plt.title(cls)
            plt.axis("off")
            idx += 1
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "sample_images_per_class.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[SAVED] {path}")

def plot_image_resolution_check(df: pd.DataFrame):
    widths, heights = [], []
    for path in df["Path"].sample(min(100, len(df))):
        img = cv2.imread(path)
        if img is not None:
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)
    plt.figure(figsize=(6, 4))
    plt.scatter(widths, heights, alpha=0.6, color="purple")
    plt.xlabel("Width (px)")
    plt.ylabel("Height (px)")
    plt.title("Image Resolution Distribution")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "image_resolution_distribution.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[SAVED] {path}")

def visualize_augmentation_examples(generator, n: int = 5):
    X, _ = generator.__getitem__(0)
    plt.figure(figsize=(n * 3, 3))
    for i in range(min(n, len(X))):
        img = X[i]
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        plt.subplot(1, n, i + 1)
        plt.imshow(img)
        plt.title("Augmented")
        plt.axis("off")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "augmentation_examples.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[SAVED] {path}")

# =====================================================================
# 7. PDF REPORT BUILDER
# =====================================================================
def build_pdf_report():
    """
    Combines all generated .png plots into a single PDF summary.
    """
    pdf_path = os.path.join(RESULTS_DIR, "Data_Report.pdf")
    with PdfPages(pdf_path) as pdf:
        for img_file in sorted(os.listdir(RESULTS_DIR)):
            if img_file.endswith(".png"):
                img_path = os.path.join(RESULTS_DIR, img_file)
                fig = plt.figure(figsize=(8, 6))
                plt.imshow(plt.imread(img_path))
                plt.axis("off")
                plt.title(img_file.replace("_", " ").replace(".png", ""), fontsize=12)
                pdf.savefig(fig)
                plt.close(fig)
    print(f"📄 [REPORT SAVED] {pdf_path}")

# =====================================================================
# 8. MAIN TEST
# =====================================================================
if __name__ == "__main__":
    DATA_DIR = "Dataset_BUSI_with_GT"

    train_df, val_df, test_df = prepare_data_frames(DATA_DIR, balance=True)

    plot_image_resolution_check(train_df)
    visualize_sample_images(train_df)

    gen = MedicalDataGenerator(train_df, augment=True)
    visualize_augmentation_examples(gen)

    build_pdf_report()
