"""
config.py

Central configuration for the RA-TKD project: project/output directories,
dataset CSV and image archive paths, and shared training hyperparameters
used across the DenseNet161 teacher and RA-TKD student training/evaluation
scripts.

Author: Mohammed Shimal Iddin
Student Number: 40490740
Module: ECS8056: Themed Research Project (2251_SUM)
Supervisor: Dr. Muhammad Fahim
Last Updated: 2026-08-31
"""

from pathlib import Path
import torch



# Project directories
PROJECT_ROOT = Path(r"/home/40490740@eeecs.qub.ac.uk/Downloads/RA-TKD-DeiT")
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"



# Dataset files and image archives
OLD_PROJECT_DATA_DIR = Path(r"/home/40490740@eeecs.qub.ac.uk/Downloads/archive")
TRAIN_CSV = OLD_PROJECT_DATA_DIR / "train.csv"
VAL_CSV = OLD_PROJECT_DATA_DIR / "val.csv"
TEST_CSV = OLD_PROJECT_DATA_DIR / "test.csv"


# The dataset searches image archives images_001 to images_012 under this directory.
IMAGE_ROOT = OLD_PROJECT_DATA_DIR



# Training configuration
NUM_CLASSES = 14
IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"