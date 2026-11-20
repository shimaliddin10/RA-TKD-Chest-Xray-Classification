"""
eval_student_densenet_augmented.py

Test-set evaluation script for the best-performing RA-TKD student
configuration: clean loss pipeline with alpha = 0.7, trained with
training-time data augmentation, cosine learning-rate annealing, and
AUROC-based checkpoint selection, distilled from the DenseNet161 teacher.

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
    # Deterministic preprocessing used for held-out test evaluation.
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Create a non-shuffled test loader for deterministic evaluation.
    test_ds = ChestXrayDataset(TEST_CSV, IMAGE_ROOT, transform=get_transforms())
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )


    # Load the checkpoint of the augmented RA-TKD configuration.
    model = RATKDDeiT(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(
        torch.load(
            CHECKPOINT_DIR / "ratkd_deit_densenet_augmented.pth",
            map_location=device
        )
    )
    model.eval()


    all_targets, all_probs = [], []


    # Run inference without gradients and collect predictions for the full test set.
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(labels.numpy())


    y_true = np.concatenate(all_targets, axis=0)
    y_score = np.concatenate(all_probs, axis=0)


    # Macro AUROC weights every pathology equally despite class imbalance.
    macro_auc = roc_auc_score(y_true, y_score, average="macro")
    print(f"RA-TKD-DeiT (DenseNet teacher) Macro AUROC: {macro_auc:.4f}")


    print("\nPer-class AUROC:")
    per_class_results = {}


    # Compute AUROC per pathology if both positive and negative examples are present.
    for i, class_name in enumerate(LABEL_COLS):
        if len(np.unique(y_true[:, i])) > 1:
            class_auc = roc_auc_score(y_true[:, i], y_score[:, i])
        else:
            class_auc = float("nan")


        per_class_results[class_name] = class_auc
        print(f"  {class_name:<20}: {class_auc:.4f}")


    # Save results for subsequent tables and per-pathology performance plots.
    df = pd.DataFrame(
        list(per_class_results.items()),
        columns=["Class", "AUROC"]
    )
    df.to_csv("ratkd_deit_augmented_per_class_auroc.csv", index=False)
    print("\nSaved results to ratkd_deit_augmented_per_class_auroc.csv")



if __name__ == "__main__":
    main()