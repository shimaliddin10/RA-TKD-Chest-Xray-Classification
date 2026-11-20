"""
fix_train_val_leakage.py

Fixes train/val patient-level leakage found by check_patient_leakage.py.

Confirmed situation:
  Train/Val overlap:  4574 patients  <-- problem
  Train/Test overlap: 0 patients     <-- already clean, untouched
  Val/Test overlap:    0 patients     <-- already clean, untouched

Strategy: move all rows belonging to the overlapping patients from val
into train. This leaves test completely untouched (it was already clean)
and makes val exclusively patients never seen during training.

Backs up the original train.csv/val.csv before overwriting.

Author: Mohammed Shimal Iddin
Student Number: 40490740
Module: ECS8056: Themed Research Project (2251_SUM)
Supervisor: Dr. Muhammad Fahim
Last Updated: 2026-08-31
"""


import pandas as pd
import shutil
from pathlib import Path


# matching my config.py OLD_PROJECT_DATA_DIR
PROJECT_DATA_DIR = Path(r"/home/40490740@eeecs.qub.ac.uk/Downloads/archive")


TRAIN_CSV = PROJECT_DATA_DIR / "train.csv"
VAL_CSV = PROJECT_DATA_DIR / "val.csv"
TEST_CSV = PROJECT_DATA_DIR / "test.csv"



def load(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    df["patient_id"] = df["image_name"].astype(str).str.split("_").str[0]
    return df



def main():
    train_df = load(TRAIN_CSV)
    val_df = load(VAL_CSV)
    test_df = load(TEST_CSV)


    train_ids = set(train_df["patient_id"])
    val_ids = set(val_df["patient_id"])
    test_ids = set(test_df["patient_id"])


    overlap_ids = train_ids & val_ids
    print(f"Found {len(overlap_ids)} overlapping train/val patients.")


    # sanity re-check against test just in case
    assert len(train_ids & test_ids) == 0, "Train/Test overlap found - stop and investigate."
    assert len(val_ids & test_ids) == 0, "Val/Test overlap found - stop and investigate."


    # split val rows into "move to train" vs "keep in val"
    move_mask = val_df["patient_id"].isin(overlap_ids)
    val_rows_to_move = val_df[move_mask].drop(columns=["patient_id"])
    val_rows_to_keep = val_df[~move_mask].drop(columns=["patient_id"])


    new_train_df = pd.concat([train_df.drop(columns=["patient_id"]), val_rows_to_move], ignore_index=True)
    new_val_df = val_rows_to_keep


    print(f"Moved {len(val_rows_to_move)} images from val into train.")
    print(f"New train size: {len(new_train_df)} (was {len(train_df)})")
    print(f"New val size:   {len(new_val_df)} (was {len(val_df)})")


    # backup originals
    shutil.copy(TRAIN_CSV, TRAIN_CSV.with_suffix(".csv.bak"))
    shutil.copy(VAL_CSV, VAL_CSV.with_suffix(".csv.bak"))
    print(f"\nBacked up originals to {TRAIN_CSV.with_suffix('.csv.bak')} and {VAL_CSV.with_suffix('.csv.bak')}")


    new_train_df.to_csv(TRAIN_CSV, index=False)
    new_val_df.to_csv(VAL_CSV, index=False)
    print("Wrote cleaned train.csv and val.csv. test.csv left untouched.")


    # final verification
    final_train_ids = set(new_train_df["image_name"].astype(str).str.split("_").str[0])
    final_val_ids = set(new_val_df["image_name"].astype(str).str.split("_").str[0])
    assert len(final_train_ids & final_val_ids) == 0, "Leakage still present after fix - investigate."
    print("Verified: 0 patient overlap between new train and new val.")



if __name__ == "__main__":
    main()