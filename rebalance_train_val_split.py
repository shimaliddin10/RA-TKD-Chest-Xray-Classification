"""
Rebalances train/val after leakage fix.


Context: fix_train_val_leakage.py correctly removed train/val patient
overlap, but left val very small (1705 images / 1664 patients out of
112,120 total) - risky for rare classes like Hernia.


This script pools the current train.csv + val.csv together by patient,
then re-splits at ~90/10 using patient-level grouping (never splitting
a patient's images across train/val). test.csv is never touched.


Backs up current train.csv/val.csv (post-leakage-fix versions) before
overwriting, so you can always get back to this exact state.


Author: Mohammed Shimal Iddin
Student Number: 40490740
Module: ECS8056: Themed Research Project (2251_SUM)
Supervisor: Dr. Muhammad Fahim
Last Updated: 2026-08-31
"""


import pandas as pd
import numpy as np
import shutil
from pathlib import Path


# matching my config.py OLD_PROJECT_DATA_DIR
PROJECT_DATA_DIR = Path(r"/home/40490740@eeecs.qub.ac.uk/Downloads/archive")


TRAIN_CSV = PROJECT_DATA_DIR / "train.csv"
VAL_CSV = PROJECT_DATA_DIR / "val.csv"
TEST_CSV = PROJECT_DATA_DIR / "test.csv"


VAL_FRACTION = 0.10   # target fraction of pooled train+val patients for val
SEED = 42



def load(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    df["patient_id"] = df["image_name"].astype(str).str.split("_").str[0]
    return df



def main():
    rng = np.random.default_rng(SEED)


    train_df = load(TRAIN_CSV)
    val_df = load(VAL_CSV)
    test_df = load(TEST_CSV)


    label_cols = [c for c in train_df.columns if c not in ("image_name", "patient_id")]


    pooled = pd.concat([train_df, val_df], ignore_index=True)
    test_ids = set(test_df["image_name"].astype(str).str.split("_").str[0])


    pooled_ids = pooled["patient_id"].unique()
    assert len(set(pooled_ids) & test_ids) == 0, "Pooled train+val overlaps test - stop and investigate."


    rng.shuffle(pooled_ids)
    n_val_patients = int(len(pooled_ids) * VAL_FRACTION)
    val_patient_ids = set(pooled_ids[:n_val_patients])
    train_patient_ids = set(pooled_ids[n_val_patients:])


    new_val_df = pooled[pooled["patient_id"].isin(val_patient_ids)].drop(columns=["patient_id"])
    new_train_df = pooled[pooled["patient_id"].isin(train_patient_ids)].drop(columns=["patient_id"])


    print(f"Pooled patients: {len(pooled_ids)}")
    print(f"New train: {len(new_train_df)} images, {len(train_patient_ids)} patients")
    print(f"New val:   {len(new_val_df)} images, {len(val_patient_ids)} patients")
    print()


    # per-class positive counts in new val, to check rare-class coverage
    print("Per-class positive counts in new val set:")
    for c in label_cols:
        pos = int(new_val_df[c].sum())
        print(f"  {c}: {pos}")
    print()


    low_coverage = [c for c in label_cols if new_val_df[c].sum() < 10]
    if low_coverage:
        print(f"WARNING: these classes have fewer than 10 positive val examples: {low_coverage}")
        print("Per-class AUROC for these will be noisy. Consider a larger VAL_FRACTION if this matters.")


    # backup current (post-leakage-fix) train/val before overwriting
    shutil.copy(TRAIN_CSV, TRAIN_CSV.with_suffix(".csv.prerebalance.bak"))
    shutil.copy(VAL_CSV, VAL_CSV.with_suffix(".csv.prerebalance.bak"))
    print(f"Backed up pre-rebalance train/val to *.csv.prerebalance.bak")


    new_train_df.to_csv(TRAIN_CSV, index=False)
    new_val_df.to_csv(VAL_CSV, index=False)
    print("Wrote rebalanced train.csv and val.csv. test.csv left untouched.")


    # final verification: no patient overlap anywhere
    final_train_ids = set(new_train_df["image_name"].astype(str).str.split("_").str[0])
    final_val_ids = set(new_val_df["image_name"].astype(str).str.split("_").str[0])
    assert len(final_train_ids & final_val_ids) == 0
    assert len(final_train_ids & test_ids) == 0
    assert len(final_val_ids & test_ids) == 0
    print("Verified: 0 patient overlap across train/val/test.")



if __name__ == "__main__":
    main()