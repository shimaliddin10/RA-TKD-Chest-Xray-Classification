"""
dataset.py

PyTorch Dataset for NIH ChestX-ray14 multi-label chest X-ray classification.

Loads image filenames and their 14-dimensional pathology-label vectors from a
CSV file, resolving each filename against the 12 NIH image archive folders
(images_001 .. images_012). Each label column is treated as an independent
binary target, consistent with the project's multi-label formulation.

Author: Mohammed Shimal Iddin
Student Number: 40490740
Module: ECS8056: Themed Research Project (2251_SUM)
Supervisor: Dr. Muhammad Fahim
Last Updated: 2026-08-31
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path



class ChestXrayDataset(Dataset):
    """Loads chest X-ray images and their pathology-label vectors from a CSV file."""


    def __init__(self, csv_path, image_root, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform


        # Standardise CSV column names and identify the image-filename column.
        self.df.columns = [str(c).strip() for c in self.df.columns]
        self.image_col = "image_name"


        if self.image_col not in self.df.columns:
            raise ValueError(
                f"Expected column '{self.image_col}' in {csv_path}, "
                f"got {self.df.columns.tolist()}"
            )


        # Every column except the image filename is treated as a pathology label.
        self.label_cols = [c for c in self.df.columns if c != self.image_col]


        base = Path(image_root)


        # NIH ChestX-ray image archives searched when resolving an image filename.
        self.image_dirs = [
            base / "images_001" / "images",
            base / "images_002" / "images",
            base / "images_003" / "images",
            base / "images_004" / "images",
            base / "images_005" / "images",
            base / "images_006" / "images",
            base / "images_007" / "images",
            base / "images_008" / "images",
            base / "images_009" / "images",
            base / "images_010" / "images",
            base / "images_011" / "images",
            base / "images_012" / "images",
        ]


        # Print dataset configuration to verify paths and CSV fields at initialisation.
        print(f"[Dataset] CSV: {csv_path}")
        print(f"[Dataset] Image column: {self.image_col}")
        print(f"[Dataset] Label columns: {self.label_cols}")
        print(f"[Dataset] Image dirs:")
        for d in self.image_dirs:
            print(f"  - {d}")


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row[self.image_col]).strip()


        # Locate the image across the 12 NIH image archive folders.
        img_path = None
        for d in self.image_dirs:
            candidate = d / img_name
            if candidate.exists():
                img_path = candidate
                break


        if img_path is None:
            raise FileNotFoundError(
                f"Image {img_name} not found in any images_XXX\\images folder"
            )


        # Convert to RGB so every model receives a three-channel input image.
        image = Image.open(img_path).convert("RGB")
        labels = torch.tensor(row[self.label_cols].values.astype("float32"))


        if self.transform:
            image = self.transform(image)


        return image, labels