"""
train_student.py

Distributed (multi-GPU, DDP) training script for the RA-TKD-DeiT student,
distilling from a frozen ViT teacher (ViTTeacherWithFeatures,
teacher_dim=768) rather than the DenseNet161 teacher used in the other
train_student_densenet*.py scripts.

Uses NCCL-based DistributedDataParallel with a DistributedSampler for
both splits, automatic mixed precision (autocast + GradScaler), and
imports RATKDDeiT from the base novel_model.py (fixed default alpha =
0.3 inside RegionAwareTokenSelector). Loss functions (MultiLabelKDLoss,
FeatureDistillationLoss) are imported from novel_losses.py here, with a
different constructor/call signature than the losses.py versions used
elsewhere in this project. Tracks a world-size-averaged validation macro
AUROC each epoch and checkpoints the unwrapped student state to
ratkd_deit_best_auc.pth from the rank-0 process only.

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
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from torch.amp import autocast, GradScaler


from config import *
from dataset import ChestXrayDataset
from model_teacher import ViTTeacherWithFeatures
from novel_model import RATKDDeiT
from novel_losses import MultiLabelKDLoss, FeatureDistillationLoss



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



def setup_ddp():
    # Initialise NCCL-based distributed training and assign this process to its GPU.
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank



def cleanup_ddp():
    # Release distributed-process resources after training.
    dist.destroy_process_group()



def save_checkpoint(state_dict, path):
    # Save the model state dictionary from the main process.
    torch.save(state_dict, path)



def main():
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")


    # Only rank zero prints logs and saves checkpoints.
    is_main = dist.get_rank() == 0


    checkpoint_dir = Path(CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)


    # Load training and validation splits.
    train_ds = ChestXrayDataset(TRAIN_CSV, IMAGE_ROOT, transform=get_transforms())
    val_ds = ChestXrayDataset(VAL_CSV, IMAGE_ROOT, transform=get_transforms())


    # Split dataset samples across distributed processes.
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)


    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=False
    )


    # Load the ViT teacher and keep its parameters fixed.
    teacher = ViTTeacherWithFeatures(num_classes=NUM_CLASSES).to(device)
    teacher_ckpt = torch.load(
        checkpoint_dir / "vit_teacher_best_auc.pth",
        map_location=device
    )
    teacher.load_state_dict(teacher_ckpt)
    teacher.eval()


    for p in teacher.parameters():
        p.requires_grad = False


    # Wrap the RA-TKD student in DistributedDataParallel for multi-GPU training.
    student = RATKDDeiT(num_classes=NUM_CLASSES).to(device)
    student = DDP(
        student,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )


    optimizer = optim.AdamW(
        student.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS
    )


    # Use logit and feature distillation from the ViT teacher.
    kd_criterion = MultiLabelKDLoss(
        alpha_hard=1.0,
        alpha_soft=0.7,
        temperature=4.0
    ).to(device)
    feat_criterion = FeatureDistillationLoss(
        student_dim=192,
        teacher_dim=768
    ).to(device)


    # Mixed-precision scaler for efficient GPU training.
    scaler = GradScaler("cuda")


    best_auc = -1.0


    for epoch in range(NUM_EPOCHS):
        # Reshuffle distributed training samples consistently for this epoch.
        train_sampler.set_epoch(epoch)


        student.train()
        train_loss = 0.0


        # Train the DDP student using frozen ViT-teacher supervision.
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()


            optimizer.zero_grad(set_to_none=True)


            with torch.no_grad():
                t_out = teacher(images, return_features=True)


            with autocast("cuda"):
                s_out = student(images, return_features=True)
                kd_total, kd_stats = kd_criterion(
                    s_out["logits"],
                    t_out["logits"],
                    labels
                )
                feat_loss = feat_criterion(
                    s_out["features"],
                    t_out["features"]
                )
                loss = kd_total + 0.3 * feat_loss


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            train_loss += loss.detach().item()


        scheduler.step()


        student.eval()
        val_loss = 0.0
        all_probs = []
        all_labels = []


        # Evaluate the model on each process's validation subset.
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).float()


                with autocast("cuda"):
                    t_out = teacher(images, return_features=True)
                    s_out = student(images, return_features=True)


                    kd_total, _ = kd_criterion(
                        s_out["logits"],
                        t_out["logits"],
                        labels
                    )
                    feat_loss = feat_criterion(
                        s_out["features"],
                        t_out["features"]
                    )
                    loss = kd_total + 0.3 * feat_loss


                probs = torch.sigmoid(s_out["logits"])
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


            # Save the unwrapped student state when validation macro AUROC improves.
            if global_auc > best_auc:
                best_auc = global_auc
                save_checkpoint(
                    student.module.state_dict(),
                    checkpoint_dir / "ratkd_deit_best_auc.pth"
                )
                print(
                    f"Saved best student checkpoint with mAUROC={best_auc:.4f}"
                )


    cleanup_ddp()



if __name__ == "__main__":
    main()