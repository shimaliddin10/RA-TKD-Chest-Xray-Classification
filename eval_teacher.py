"""
eval_teacher.py

Test-set evaluation script for the ViT-B/16 teacher model.

Loads the ViT teacher checkpoint selected by validation macro AUROC, runs
inference over the held-out test split with pinned memory for efficient
CUDA transfer, and reports macro AUROC and per-pathology AUROC across the
14 independent multi-label thoracic-finding outputs.

Author: Mohammed Shimal Iddin
Student Number: 40490740
Module: ECS8056: Themed Research Project (2251_SUM)
Supervisor: Dr. Muhammad Fahim
Last Updated: 2026-08-31
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score


from config import *
from dataset import ChestXrayDataset
from model_teacher import ViTTeacherWithFeatures



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


    # Create the test loader with pinned memory for efficient CUDA transfer.
    test_ds = ChestXrayDataset(TEST_CSV, IMAGE_ROOT, transform=get_transforms())
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )


    # Load the ViT teacher checkpoint selected by validation macro AUROC.
    model = ViTTeacherWithFeatures(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(
        torch.load(
            CHECKPOINT_DIR / "vit_teacher_best_auc.pth",
            map_location=device
        )
    )
    model.eval()


    all_targets, all_probs = [], []


    # Run inference without gradients and collect predictions across the test set.
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()


            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()


            all_probs.append(probs)
            all_targets.append(labels.cpu().numpy())


    y_true = np.concatenate(all_targets, axis=0)
    y_score = np.concatenate(all_probs, axis=0)


    # Macro AUROC weights each pathology output equally.
    macro_auc = roc_auc_score(y_true, y_score, average="macro")
    print(f"Teacher Macro AUROC: {macro_auc:.4f}")


    print("\nPer-class AUROC:")
    per_class_results = {}


    # Compute AUROC per pathology when both positive and negative cases exist.
    for i, name in enumerate(test_ds.label_cols):
        if len(np.unique(y_true[:, i])) < 2:
            print(f"  {name:20s}: N/A (only one class present)")
            per_class_results[name] = None
            continue


        auc = roc_auc_score(y_true[:, i], y_score[:, i])
        per_class_results[name] = auc
        print(f"  {name:20s}: {auc:.4f}")


    # Save macro and per-pathology AUROC values for reporting.
    with open("teacher_per_class_auroc.csv", "w") as f:
        f.write("class,auroc\n")
        f.write(f"macro,{macro_auc:.4f}\n")
        for name, auc in per_class_results.items():
            f.write(f"{name},{auc if auc is not None else 'NA'}\n")


    print("\nSaved results to teacher_per_class_auroc.csv")



if __name__ == "__main__":
    main()