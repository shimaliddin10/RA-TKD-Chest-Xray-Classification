"""
train_teacher_densenet.py

Trains the DenseNet161 teacher (DenseNet161TeacherWithFeatures) for
multi-label chest X-ray pathology classification, using focal binary
cross-entropy hard-label supervision only (no distillation, since this
is the teacher itself).

Uses shared deterministic preprocessing for training and validation,
FocalBCELoss (gamma=2.0) as the objective, AdamW optimisation, and
automatic mixed precision via torch.amp autocast/GradScaler. Model
selection and early stopping are driven by validation loss rather than
AUROC: the checkpoint with the lowest validation loss (improvement
threshold MIN_DELTA=1e-4) is saved to densenet_teacher_best.pth, with
training stopped after 5 epochs without a meaningful improvement. This
checkpoint is the frozen teacher loaded by every train_student_densenet*.py
script.

Author: Mohammed Shimal Iddin
Student Number: 40490740
Module: ECS8056: Themed Research Project (2251_SUM)
Supervisor: Dr. Muhammad Fahim
Last Updated: 2026-08-31
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.amp import autocast, GradScaler


from config import *
from dataset import ChestXrayDataset
from model_teacher_densenet import DenseNet161TeacherWithFeatures
from losses import FocalBCELoss



def get_transforms():
    # Shared resize and ImageNet normalisation for training and validation.
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])



def main():
    # Ensure checkpoint storage exists before training begins.
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Load the training and validation splits.
    train_ds = ChestXrayDataset(TRAIN_CSV, IMAGE_ROOT, transform=get_transforms())
    val_ds = ChestXrayDataset(VAL_CSV, IMAGE_ROOT, transform=get_transforms())


    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )


    # Train a DenseNet161 teacher using focal hard-label supervision.
    model = DenseNet161TeacherWithFeatures(num_classes=NUM_CLASSES).to(device)
    criterion = FocalBCELoss(gamma=2.0)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )


    # Enable automatic mixed precision for GPU training.
    scaler = GradScaler("cuda")


    PATIENCE = 5
    MIN_DELTA = 1e-4


    best_val = float("inf")
    epochs_no_improve = 0


    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0


        # Update teacher parameters using the training split.
        for images, labels in train_loader:
            images, labels = (
                images.to(device, non_blocking=True),
                labels.to(device, non_blocking=True)
            )


            optimizer.zero_grad()


            with autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, labels)


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            train_loss += loss.item()


        model.eval()
        val_loss = 0.0


        # Evaluate validation loss without gradient calculations.
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = (
                    images.to(device, non_blocking=True),
                    labels.to(device, non_blocking=True)
                )


                with autocast("cuda"):
                    logits = model(images)
                    loss = criterion(logits, labels)


                val_loss += loss.item()


        train_loss /= len(train_loader)
        val_loss /= len(val_loader)


        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f}"
        )


        # Save the checkpoint with the lowest validation loss.
        if val_loss < best_val - MIN_DELTA:
            best_val = val_loss
            epochs_no_improve = 0


            torch.save(
                model.state_dict(),
                CHECKPOINT_DIR / "densenet_teacher_best.pth"
            )
            print(
                f"Saved best DenseNet teacher checkpoint "
                f"(val_loss={val_loss:.4f})"
            )


        else:
            epochs_no_improve += 1
            print(
                f"No improvement for {epochs_no_improve} epoch(s) "
                f"(best={best_val:.4f})"
            )


        # Stop after five epochs without a meaningful validation-loss improvement.
        if epochs_no_improve >= PATIENCE:
            print(
                f"Early stopping triggered at epoch {epoch+1}. "
                f"Best val_loss={best_val:.4f}"
            )
            break


    print(
        f"Training finished. Best checkpoint saved with val_loss={best_val:.4f}"
    )



if __name__ == "__main__":
    main()