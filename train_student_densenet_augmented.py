"""
Trains the RA-TKD-DeiT student via knowledge distillation from a frozen
DenseNet161 teacher, using training-time data augmentation.

Unlike the alpha00/alpha03/alpha07 clean-training scripts, this pipeline
applies random rotation, random-resized cropping, and colour jitter to
training images (validation preprocessing remains deterministic), and
imports RATKDDeiT from the base novel_model.py (fixed default alpha =
0.3 inside RegionAwareTokenSelector, not exposed as a constructor
argument). Combines the same multi-label hard-label/logit-KD loss and
feature-distillation loss as the clean scripts. Tracks macro AUROC on
the validation split each epoch (also logging the selector's live alpha
value), checkpoints the best model to
ratkd_deit_densenet_augmented.pth, and early-stops after 10 epochs
without improvement. This is the augmentation counterpart referenced in
eval_student_densenet_augmented.py, and underlies the project's
best-performing student result (test AUROC 0.7672).

Author: Mohammed Shimal Iddin
Student Number: 40490740
Module: ECS8056: Themed Research Project (2251_SUM)
Supervisor: Dr. Muhammad Fahim
Last Updated: 2026-08-31
"""

import os
import numpy as np
from sklearn.metrics import roc_auc_score


import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms


from config import *


from dataset import ChestXrayDataset
from model_teacher_densenet import DenseNet161TeacherWithFeatures
from novel_model import RATKDDeiT
from losses import MultiLabelKDLoss, FeatureDistillationLoss  # Clean KD and feature-distillation losses.



def get_train_transforms():
    # Training-only augmentation followed by ImageNet normalisation.
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])



def get_eval_transforms():
    # Deterministic preprocessing used for validation.
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])



def compute_macro_auc(y_true, y_prob):
    # Average valid per-pathology AUROC values.
    aucs = []


    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) > 1:
            aucs.append(roc_auc_score(y_true[:, i], y_prob[:, i]))


    return float(np.mean(aucs)) if len(aucs) > 0 else 0.0



def get_alpha_str(student):
    """Returns the current selector alpha value for training logs."""
    alpha_val = student.selector.alpha


    if torch.is_tensor(alpha_val):
        return f"{alpha_val.item():.4f}"


    return f"{alpha_val:.4f}"



def main():
    # Ensure checkpoint storage exists before training.
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Apply augmentation only to training images; keep validation preprocessing fixed.
    train_ds = ChestXrayDataset(
        TRAIN_CSV,
        IMAGE_ROOT,
        transform=get_train_transforms()
    )
    val_ds = ChestXrayDataset(
        VAL_CSV,
        IMAGE_ROOT,
        transform=get_eval_transforms()
    )


    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )


    # Load and freeze the pretrained DenseNet161 teacher.
    teacher = DenseNet161TeacherWithFeatures(num_classes=NUM_CLASSES).to(device)
    teacher.load_state_dict(
        torch.load(
            CHECKPOINT_DIR / "densenet_teacher_best.pth",
            map_location=device
        )
    )
    teacher.eval()


    for p in teacher.parameters():
        p.requires_grad = False


    # Initialise the RA-TKD student.
    student = RATKDDeiT(num_classes=NUM_CLASSES).to(device)


    optimizer = optim.AdamW(
        student.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS
    )


    # Use the clean hard-label, logit-KD, and feature-distillation objective.
    kd_criterion = MultiLabelKDLoss(
        alpha_bce=1.0,
        alpha_kd=0.7,
        temperature=3.0
    ).to(device)


    feat_criterion = FeatureDistillationLoss(
        student_dim=192,
        teacher_mid_dim=384,
        teacher_final_dim=2208
    ).to(device)


    best_auc = -1.0
    patience = 10
    epochs_no_improve = 0


    for epoch in range(NUM_EPOCHS):
        student.train()
        train_loss = 0.0


        # Train using augmented inputs and frozen teacher supervision.
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()


            with torch.no_grad():
                t_out = teacher(images, return_features=True)


            s_out = student(images, return_features=True)


            kd_total, hard_loss, kd_loss = kd_criterion(
                s_out["logits"],
                t_out["logits"],
                labels
            )
            feat_loss = feat_criterion(
                s_out["features"],
                t_out["features"]
            )
            loss = kd_total + 0.3 * feat_loss


            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


            train_loss += loss.item()


        scheduler.step()


        student.eval()
        val_loss = 0.0
        all_probs = []
        all_labels = []


        # Evaluate on unaugmented validation images.
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).float()


                t_out = teacher(images, return_features=True)
                s_out = student(images, return_features=True)


                kd_total, _, _ = kd_criterion(
                    s_out["logits"],
                    t_out["logits"],
                    labels
                )
                feat_loss = feat_criterion(
                    s_out["features"],
                    t_out["features"]
                )
                loss = kd_total + 0.3 * feat_loss
                val_loss += loss.item()


                probs = torch.sigmoid(s_out["logits"])
                all_probs.append(probs.detach().cpu())
                all_labels.append(labels.detach().cpu())


        train_loss /= len(train_loader)
        val_loss /= len(val_loader)


        all_probs = torch.cat(all_probs, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        val_auc = compute_macro_auc(all_labels, all_probs)


        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"Val mAUROC: {val_auc:.4f} | "
            f"alpha: {get_alpha_str(student)}"
        )


        # Save the checkpoint with the highest validation macro AUROC.
        if val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0


            torch.save(
                student.state_dict(),
                CHECKPOINT_DIR / "ratkd_deit_densenet_augmented.pth"
            )
            print(
                f"Saved best student checkpoint (augmented), "
                f"mAUROC={best_auc:.4f}"
            )


        else:
            epochs_no_improve += 1


            # Stop when validation macro AUROC has not improved for ten epochs.
            if epochs_no_improve >= patience:
                print(
                    f"Early stopping at epoch {epoch+1} "
                    f"(no improvement for {patience} epochs). "
                    f"Best mAUROC={best_auc:.4f}"
                )
                break



if __name__ == "__main__":
    main()