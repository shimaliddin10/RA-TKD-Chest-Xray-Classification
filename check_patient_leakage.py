"""
check_patient_leakage.py

Patient-level leakage check for train/val/test CSVs.

NIH ChestX-ray14 filenames encode the patient ID as the first 8 digits,
e.g. "00000001_000.png" -> patient ID "00000001".
This script checks whether any patient appears in more than one split.

Run this from the same environment as your project (it just needs pandas),
pointing PROJECT_DATA_DIR at the same folder your config.py uses:
    OLD_PROJECT_DATA_DIR = ".../MSc/cxr-vit-kd/data"

Author: Mohammed Shimal Iddin
Student Number: 40490740
Module: ECS8056: Themed Research Project (2251_SUM)
Supervisor: Dr. Muhammad Fahim
Last Updated: 2026-08-31
"""


import pandas as pd
from pathlib import Path


# EDIT THIS to match your config.py OLD_PROJECT_DATA_DIR
PROJECT_DATA_DIR = Path(r"/home/40490740@eeecs.qub.ac.uk/Downloads/archive")


TRAIN_CSV = PROJECT_DATA_DIR / "train.csv"
VAL_CSV = PROJECT_DATA_DIR / "val.csv"
TEST_CSV = PROJECT_DATA_DIR / "test.csv"



def get_patient_ids(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    if "image_name" not in df.columns:
        raise ValueError(f"Expected 'image_name' column in {csv_path}, got {df.columns.tolist()}")
    # NIH ChestX-ray14 convention: patient ID = first 8 chars before the underscore
    patient_ids = df["image_name"].astype(str).str.split("_").str[0]
    return set(patient_ids), len(df)



def main():
    train_ids, n_train = get_patient_ids(TRAIN_CSV)
    val_ids, n_val = get_patient_ids(VAL_CSV)
    test_ids, n_test = get_patient_ids(TEST_CSV)


    print(f"Train: {n_train} images, {len(train_ids)} unique patients")
    print(f"Val:   {n_val} images, {len(val_ids)} unique patients")
    print(f"Test:  {n_test} images, {len(test_ids)} unique patients")
    print()


    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids


    print(f"Train/Val patient overlap:  {len(train_val_overlap)} patients")
    print(f"Train/Test patient overlap: {len(train_test_overlap)} patients")
    print(f"Val/Test patient overlap:   {len(val_test_overlap)} patients")
    print()


    total_overlap = len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap)
    if total_overlap == 0:
        print("PASS: No patient-level leakage detected. Splits are clean.")
    else:
        print("WARNING: Patient-level leakage detected!")
        print("Some patients appear in more than one split.")
        print("This will inflate val/test AUROC and make results unreliable.")
        print("Recommended fix: re-split by unique patient ID, ensuring all")
        print("images from a given patient go into exactly one of train/val/test.")


        if train_test_overlap:
            print(f"\nExample overlapping train/test patient IDs (up to 10):")
            print(list(train_test_overlap)[:10])



if __name__ == "__main__":
    main()