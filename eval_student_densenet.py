"""
eval_student_densenet.py

Test-set evaluation script for the RA-TKD clean-baseline student
configuration (score-head fix, original clean loss pipeline), distilled
from the DenseNet161 teacher.

Loads the trained checkpoint, runs inference over the held-out test split,
and reports macro AUROC and per-pathology AUROC across the 14 independent
multi-label thoracic-finding outputs.

Author: Mohammed Shimal Iddin
Student Number: 40490740
Module: ECS8056: Themed Research Project (2251_SUM)
Supervisor: Dr. Muhammad Fahim
Last Updated: 2026-08-31
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score


from config import *
from dataset import ChestXrayDataset
from novel_model import RATKDDeiT



# Pathology labels in the same order as the dataset target columns.
LABEL_COLS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
    "Fibrosis", "Pleural_Thickening", "Hernia",
]



def get_transforms():
    # Deterministic preprocessing used for test-set evaluation.
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Create the held-out test loader without shuffling.
    test_ds = ChestXrayDataset(TEST_CSV, IMAGE_ROOT, transform=get_transforms())
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )


    # Load the clean-baseline RA-TKD checkpoint for evaluation.
    model = RATKDDeiT(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(
        torch.load(
            CHECKPOINT_DIR / "ratkd_deit_densenet_clean_baseline.pth",
            map_location=device
        )
    )
    model.eval()


    all_targets, all_probs = [], []


    # Collect prediction probabilities and labels for the complete test set.
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(labels.numpy())


    y_true = np.concatenate(all_targets, axis=0)
    y_score = np.concatenate(all_probs, axis=0)


    # Macro AUROC gives equal weight to each pathology.
    macro_auc = roc_auc_score(y_true, y_score, average="macro")
    print(f"RA-TKD-DeiT (DenseNet teacher) Macro AUROC: {macro_auc:.4f}")


    print("\nPer-class AUROC:")
    per_class_results = {}


    # Compute AUROC per pathology when both positive and negative samples are available.
    for i, class_name in enumerate(LABEL_COLS):
        if len(np.unique(y_true[:, i])) > 1:
            class_auc = roc_auc_score(y_true[:, i], y_score[:, i])
        else:
            class_auc = float("nan")


        per_class_results[class_name] = class_auc
        print(f"  {class_name:<20}: {class_auc:.4f}")


    # Save per-pathology results for reporting and visualisation.
    df = pd.DataFrame(
        list(per_class_results.items()),
        columns=["Class", "AUROC"]
    )
    df.to_csv("ratkd_deit_clean_baseline_per_class_auroc.csv", index=False)
    print("\nSaved results to ratkd_deit_clean_baseline_per_class_auroc.csv")



if __name__ == "__main__":
    main()