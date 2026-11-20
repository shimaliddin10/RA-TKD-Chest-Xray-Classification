"""
eval_student_densenet_alpha03.py

Test-set evaluation script for the RA-TKD student trained with the region-aware
token selector's blend weight set to alpha = 0.3, distilled from the
DenseNet161 teacher.

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
from novel_model_alpha03 import RATKDDeiT



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


    # Evaluate on the held-out test split without shuffling or augmentation.
    test_ds = ChestXrayDataset(TEST_CSV, IMAGE_ROOT, transform=get_transforms())
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )


    # Load the trained RA-TKD checkpoint before running inference.
    model = RATKDDeiT(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(
        torch.load(
            CHECKPOINT_DIR / "ratkd_deit_densenet_alpha03.pth",
            map_location=device
        )
    )
    model.eval()


    all_targets, all_probs = [], []


    # Collect sigmoid prediction scores and ground-truth labels for the full test set.
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(labels.numpy())


    y_true = np.concatenate(all_targets, axis=0)
    y_score = np.concatenate(all_probs, axis=0)


    # Macro AUROC weights each pathology equally, regardless of prevalence.
    macro_auc = roc_auc_score(y_true, y_score, average="macro")
    print(f"RA-TKD-DeiT (DenseNet teacher) Macro AUROC: {macro_auc:.4f}")


    print("\nPer-class AUROC:")
    per_class_results = {}


    # Calculate AUROC for every pathology with both positive and negative samples.
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
    df.to_csv("ratkd_deit_alpha03_per_class_auroc.csv", index=False)
    print("\nSaved results to ratkd_deit_alpha03_per_class_auroc.csv")



if __name__ == "__main__":
    main()