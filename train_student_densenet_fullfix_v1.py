"""
train_student_densenet_fullfix_v1.py

Trains the "full fix v1" RA-TKD-DeiT student: an attention-supervised
knowledge-distillation variant that adds class-weighted hard labels and
Grad-CAM-guided attention alignment on top of the standard logit and
feature distillation losses.

Differences from the earlier clean/augmented training scripts:
  - Uses novel_losses.MultiLabelKDLoss (not losses.MultiLabelKDLoss),
    which accepts a per-class pos_weight and an alpha_attn term.
  - Computes per-class positive weights from training-label prevalence
    via compute_pos_weight, clamped to a maximum of 20.0, to counter
    class imbalance (e.g. Hernia being extremely rare).
  - Extracts Grad-CAM attention maps from the frozen DenseNet161
    teacher's final dense block (denseblock4) for a random subset of
    each batch, and aligns them against the student's region-aware
    token_scores via an attention loss term (alpha_attn = 0.3).
  - Imports RATKDDeiT from the base novel_model.py (fixed default
    alpha = 0.3 inside RegionAwareTokenSelector).

Checkpoints the best model (by validation macro AUROC) to
ratkd_deit_densenet_fullfix_v1.pth, with early stopping after 10 epochs
without improvement, and removes the Grad-CAM hooks once training ends.

Author: Mohammed Shimal Iddin
Student Number: 40490740
Module: ECS8056: Themed Research Project (2251_SUM)
Supervisor: Dr. Muhammad Fahim
Last Updated: 2026-08-31
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms


from config import *
from dataset import ChestXrayDataset
from model_teacher_densenet import DenseNet161TeacherWithFeatures
from novel_model import RATKDDeiT
from losses import FeatureDistillationLoss
from novel_losses import MultiLabelKDLoss, GradCAM, compute_pos_weight



# Attention-supervised RA-TKD variant with class weighting and Grad-CAM guidance.
LABEL_COLS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
    "Fibrosis", "Pleural_Thickening", "Hernia",
]



def get_transforms():
    # Shared resize and ImageNet normalisation for training and validation.
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



def get_teacher_attention(gradcam, teacher, images, labels, max_samples=8):
    """Generates Grad-CAM targets for a random subset of each batch."""
    B = images.size(0)
    n = min(max_samples, B)


    # Limit Grad-CAM computation to reduce per-batch memory and runtime cost.
    idx = torch.randperm(B, device=images.device)[:n]


    images_sub = images[idx].clone().requires_grad_(True)
    labels_sub = labels[idx]


    # Prefer a ground-truth positive class when available; otherwise use top prediction.
    with torch.enable_grad(), torch.amp.autocast("cuda"):
        teacher_logits = teacher(images_sub)


        if isinstance(teacher_logits, dict):
            teacher_logits = teacher_logits["logits"]


        has_pos = labels_sub.sum(dim=1) > 0
        masked_logits = teacher_logits.masked_fill(
            labels_sub == 0,
            float("-inf")
        )
        class_idx = torch.where(
            has_pos,
            masked_logits.argmax(dim=1),
            teacher_logits.argmax(dim=1),
        )


    teacher_attn = gradcam(images_sub, class_idx)


    # Release temporary tensors after Grad-CAM extraction.
    del images_sub, teacher_logits, masked_logits
    torch.cuda.empty_cache()


    return teacher_attn, idx



def main():
    # Ensure checkpoint storage exists before training.
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Load the training and validation splits.
    train_ds = ChestXrayDataset(TRAIN_CSV, IMAGE_ROOT, transform=get_transforms())
    val_ds = ChestXrayDataset(VAL_CSV, IMAGE_ROOT, transform=get_transforms())


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


    # Calculate bounded positive-class weights from training-label prevalence.
    train_df = pd.read_csv(TRAIN_CSV)
    label_counts = train_df[LABEL_COLS].sum(axis=0).values
    pos_weight = compute_pos_weight(
        label_counts,
        total_samples=len(train_df)
    ).clamp(max=20.0).to(device)


    # Load and freeze the DenseNet161 teacher.
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


    # Locate DenseNet's final dense block for Grad-CAM target extraction.
    target_layer = None


    for name, module in teacher.named_modules():
        if name.endswith("denseblock4"):
            target_layer = module
            break


    if target_layer is None:
        raise RuntimeError("Could not find 'denseblock4' in teacher model.")


    gradcam = GradCAM(teacher, target_layer=target_layer)


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


    # Combine weighted hard labels, response KD, and fixed-weight attention alignment.
    kd_criterion = MultiLabelKDLoss(
        alpha_bce=1.0,
        alpha_kd=0.7,
        alpha_attn=0.3,
        temperature=3.0,
        pos_weight=pos_weight,
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


        # Train the student using logits, features, and Grad-CAM attention targets.
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()


            with torch.no_grad():
                t_out = teacher(images, return_features=True)


            teacher_attn, attn_idx = get_teacher_attention(
                gradcam,
                teacher,
                images,
                labels
            )


            s_out = student(images, return_features=True)


            kd_total, hard_loss, kd_loss, attn_loss = kd_criterion(
                s_out["logits"],
                t_out["logits"],
                labels,
                student_attn=s_out["token_scores"][attn_idx],
                teacher_attn=teacher_attn,
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


        # Evaluate validation loss and AUROC using the same loss components.
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()


            with torch.no_grad():
                t_out = teacher(images, return_features=True)
                s_out = student(images, return_features=True)


            teacher_attn, attn_idx = get_teacher_attention(
                gradcam,
                teacher,
                images,
                labels
            )


            with torch.no_grad():
                kd_total, _, _, _ = kd_criterion(
                    s_out["logits"],
                    t_out["logits"],
                    labels,
                    student_attn=s_out["token_scores"][attn_idx],
                    teacher_attn=teacher_attn,
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
            f"attn_loss={attn_loss:.4f}"
        )


        # Save the student checkpoint with the best validation macro AUROC.
        if val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0


            torch.save(
                student.state_dict(),
                CHECKPOINT_DIR / "ratkd_deit_densenet_fullfix_v1.pth"
            )
            print(
                f"Saved best student checkpoint (full fix v1), "
                f"mAUROC={best_auc:.4f}"
            )


        else:
            epochs_no_improve += 1


            # Stop after ten consecutive validation epochs without improvement.
            if epochs_no_improve >= patience:
                print(
                    f"Early stopping at epoch {epoch+1} "
                    f"(no improvement for {patience} epochs). "
                    f"Best mAUROC={best_auc:.4f}"
                )
                break


    # Remove registered Grad-CAM hooks after training ends.
    gradcam.remove()



if __name__ == "__main__":
    main()