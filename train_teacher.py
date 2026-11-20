"""
train_teacher.py

Distributed (multi-GPU, DDP) training script for the ViT teacher
(ViTTeacherWithFeatures), trained with plain BCEWithLogitsLoss hard-label
supervision (no focal weighting, unlike the DenseNet teacher).

Uses NCCL-based DistributedDataParallel with a DistributedSampler for
both splits, automatic mixed precision (autocast + GradScaler), AdamW
with cosine annealing, and a world-size-averaged validation macro AUROC
computed each epoch via all_reduce. The unwrapped teacher state
(model.module.state_dict()) is checkpointed to vit_teacher_best_auc.pth
from the rank-0 process whenever validation macro AUROC improves. No
early stopping is implemented; training always runs the full NUM_EPOCHS.
This checkpoint is the frozen teacher loaded by train_student.py.

Author: Mohammed Shimal Iddin
Student Number: 40490740
Module: ECS8056: Themed Research Project (2251_SUM)
Supervisor: Dr. Muhammad Fahim
Last Updated: 2026-08-31
"""

import os
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score


import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from torch.amp import autocast, GradScaler


from config import *
from dataset import ChestXrayDataset
from model_teacher import ViTTeacherWithFeatures



def get_transforms():
    # Shared resize and ImageNet normalisation for training and validation.
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])



def compute_macro_auc(y_true, y_prob):
    # Average AUROC over pathology outputs that contain both target values.
    aucs = []


    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) > 1:
            aucs.append(roc_auc_score(y_true[:, i], y_prob[:, i]))


    return float(np.mean(aucs)) if len(aucs) > 0 else 0.0



def setup_ddp():
    # Initialise NCCL distributed training and assign the process to one GPU.
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank



def cleanup_ddp():
    # Release distributed-training resources after the run.
    dist.destroy_process_group()



def main():
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")


    # Only the main process writes logs and saves the best checkpoint.
    is_main = dist.get_rank() == 0


    checkpoint_dir = Path(CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)


    # Load the training and validation dataset splits.
    train_ds = ChestXrayDataset(TRAIN_CSV, IMAGE_ROOT, transform=get_transforms())
    val_ds = ChestXrayDataset(VAL_CSV, IMAGE_ROOT, transform=get_transforms())


    # Divide dataset samples across GPUs.
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)


    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )


    # Train the ViT teacher with DistributedDataParallel.
    model = ViTTeacherWithFeatures(num_classes=NUM_CLASSES).to(device)
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False
    )


    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS
    )


    # Enable automatic mixed precision for memory-efficient GPU training.
    scaler = GradScaler("cuda")


    best_auc = -1.0


    for epoch in range(NUM_EPOCHS):
        # Set epoch for reproducible distributed shuffling.
        train_sampler.set_epoch(epoch)


        model.train()
        train_loss = 0.0


        # Optimise teacher parameters over the distributed training split.
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()


            optimizer.zero_grad(set_to_none=True)


            with autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, labels)


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            train_loss += loss.detach().item()


        scheduler.step()


        model.eval()
        val_loss = 0.0
        all_probs = []
        all_labels = []


        # Evaluate each process's validation subset without gradient updates.
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).float()


                with autocast("cuda"):
                    logits = model(images)
                    loss = criterion(logits, labels)


                probs = torch.sigmoid(logits)
                val_loss += loss.detach().item()
                all_probs.append(probs.detach().cpu())
                all_labels.append(labels.detach().cpu())


        all_probs = torch.cat(all_probs, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()


        # Aggregate rank-level macro AUROC values for distributed logging.
        local_auc = compute_macro_auc(all_labels, all_probs)
        auc_tensor = torch.tensor([local_auc], device=device)
        dist.all_reduce(auc_tensor, op=dist.ReduceOp.SUM)
        global_auc = auc_tensor.item() / dist.get_world_size()


        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)


        if is_main:
            print(
                f"Epoch {epoch+1}/{NUM_EPOCHS} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val mAUROC: {global_auc:.4f}"
            )


            # Save the unwrapped ViT teacher with the best validation macro AUROC.
            if global_auc > best_auc:
                best_auc = global_auc
                torch.save(
                    model.module.state_dict(),
                    checkpoint_dir / "vit_teacher_best_auc.pth"
                )
                print(
                    f"Saved best teacher checkpoint with mAUROC={best_auc:.4f}"
                )


    cleanup_ddp()



if __name__ == "__main__":
    main()