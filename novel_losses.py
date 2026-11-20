"""
novel_losses.py

Loss functions and Grad-CAM utility for the attention-supervised RA-TKD
training pipeline.

Provides:
  - compute_pos_weight: per-class positive weighting for BCEWithLogitsLoss,
    addressing multi-label class imbalance without focal weighting.
  - AttentionAlignmentLoss: aligns the student's token attention/selector
    map against a teacher saliency map (e.g. Grad-CAM) via MSE and/or
    cosine similarity after resizing to a common spatial resolution.
  - MultiLabelKDLoss: combines hard-label BCE, temperature-scaled logit
    distillation (MSE on sigmoid outputs), and an optional attention-
    alignment term.
  - GradCAM: forward/backward hook-based Grad-CAM extractor used to derive
    the teacher saliency maps consumed by AttentionAlignmentLoss.

Author: Mohammed Shimal Iddin
Student Number: 40490740
Module: ECS8056: Themed Research Project (2251_SUM)
Supervisor: Dr. Muhammad Fahim
Last Updated: 2026-08-31
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



def compute_pos_weight(label_counts, total_samples, eps=1e-6):
    """Computes per-class positive weights for BCEWithLogitsLoss."""
    label_counts = torch.as_tensor(label_counts, dtype=torch.float32)
    neg_counts = total_samples - label_counts
    pos_weight = neg_counts / (label_counts + eps)
    return pos_weight



class AttentionAlignmentLoss(nn.Module):
    """Aligns student token attention with teacher saliency maps."""


    def __init__(self, mode="mse_cosine", target_size=(14, 14)):
        super().__init__()
        assert mode in ("mse", "cosine", "mse_cosine")
        self.mode = mode
        self.target_size = target_size


    def _prep(self, x):
        # Convert token, 2D-map, or single-channel attention inputs to a common form.
        if x.dim() == 2:
            side = int(x.shape[1] ** 0.5)
            if side * side == x.shape[1]:
                x = x.view(x.shape[0], 1, side, side)
            else:
                return x


        if x.dim() == 3:
            x = x.unsqueeze(1)


        x = F.interpolate(
            x,
            size=self.target_size,
            mode="bilinear",
            align_corners=False
        )
        x = x.flatten(1)


        # Normalise each sample to reduce scale differences between maps.
        x = x / (x.sum(dim=1, keepdim=True) + 1e-8)
        return x


    def forward(self, student_attn, teacher_attn):
        s = self._prep(student_attn)
        t = self._prep(teacher_attn).detach()


        loss = 0.0


        if self.mode in ("mse", "mse_cosine"):
            loss = loss + F.mse_loss(s, t)


        if self.mode in ("cosine", "mse_cosine"):
            cos = F.cosine_similarity(s, t, dim=1).mean()
            loss = loss + (1.0 - cos)


        return loss



class MultiLabelKDLoss(nn.Module):
    """Combines hard-label, response-distillation, and optional attention losses."""


    def __init__(
        self,
        alpha_bce=1.0,
        alpha_kd=0.7,
        alpha_attn=0.5,
        temperature=3.0,
        pos_weight=None,
        attn_mode="mse_cosine",
    ):
        super().__init__()
        self.alpha_bce = alpha_bce
        self.alpha_kd = alpha_kd
        self.alpha_attn = alpha_attn
        self.temperature = temperature
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.attn_loss_fn = AttentionAlignmentLoss(mode=attn_mode)


    def forward(
        self,
        student_logits,
        teacher_logits,
        targets,
        student_attn=None,
        teacher_attn=None,
    ):
        # Supervision from ground-truth labels.
        hard_loss = self.bce(student_logits, targets)


        # Temperature-scaled teacher and student probability matching.
        T = self.temperature
        student_soft = torch.sigmoid(student_logits / T)
        teacher_soft = torch.sigmoid(teacher_logits / T)
        kd_loss = F.mse_loss(student_soft, teacher_soft)


        total = self.alpha_bce * hard_loss + self.alpha_kd * kd_loss


        # Add attention alignment only when both attention maps are supplied.
        attn_loss_val = torch.tensor(0.0, device=student_logits.device)
        if student_attn is not None and teacher_attn is not None and self.alpha_attn > 0:
            attn_loss_val = self.attn_loss_fn(student_attn, teacher_attn)
            total = total + self.alpha_attn * attn_loss_val


        return (
            total,
            hard_loss.item(),
            kd_loss.item(),
            attn_loss_val.item() if torch.is_tensor(attn_loss_val) else attn_loss_val,
        )



class GradCAM:
    """Extracts Grad-CAM saliency maps from a selected teacher layer."""


    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None


        # Store feature activations and gradients from the target layer.
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(
            self._save_gradient
        )


    def _save_activation(self, module, inp, out):
        self.activations = out


    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]


    def __call__(self, images, class_indices):
        # Backpropagate selected-class logits to obtain Grad-CAM weights.
        self.model.zero_grad(set_to_none=True)
        logits = self.model(images)
        selected = logits.gather(1, class_indices.view(-1, 1)).sum()
        selected.backward(retain_graph=False)


        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1))
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)


        return cam.detach()


    def remove(self):
        # Remove hooks after use to prevent duplicate registrations.
        self._fwd_handle.remove()
        self._bwd_handle.remove()