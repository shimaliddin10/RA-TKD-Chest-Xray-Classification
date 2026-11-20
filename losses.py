"""
losses.py

Loss functions used to train the RA-TKD student under DenseNet161 teacher
supervision for multi-label chest X-ray classification.

Provides three components used together during distillation training:
  - FocalBCELoss: focal binary cross-entropy for the ground-truth hard-label
    signal, addressing severe multi-label class imbalance.
  - MultiLabelKDLoss: combines the focal hard-label loss with temperature-
    scaled logit distillation from the frozen teacher's soft sigmoid outputs.
  - FeatureDistillationLoss: aligns pooled, projected student features
    against detached intermediate and final teacher features via MSE.

Author: Mohammed Shimal Iddin
Student Number: 40490740
Module: ECS8056: Themed Research Project (2251_SUM)
Supervisor: Dr. Muhammad Fahim
Last Updated: 2026-08-31
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



class FocalBCELoss(nn.Module):
    """Focal binary cross-entropy loss for imbalanced independent pathology targets."""


    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma


    def forward(self, logits, targets):
        # Convert logits to independent pathology probabilities.
        p = torch.sigmoid(logits)


        # Select the predicted probability assigned to the ground-truth outcome.
        p_t = torch.where(targets == 1, p, 1 - p)


        # Avoid numerical instability when computing log probabilities.
        p_t = p_t.clamp(min=1e-6, max=1 - 1e-6)


        # Down-weight well-classified examples and focus training on harder cases.
        loss = -((1 - p_t) ** self.gamma) * torch.log(p_t)
        return loss.mean()



class MultiLabelKDLoss(nn.Module):
    """Combines focal hard-label supervision with temperature-scaled logit distillation."""


    def __init__(self, alpha_bce=1.0, alpha_kd=0.7, temperature=3.0, gamma=2.0):
        super().__init__()
        self.alpha_bce = alpha_bce
        self.alpha_kd = alpha_kd
        self.temperature = temperature
        self.bce = FocalBCELoss(gamma=gamma)


    def forward(self, student_logits, teacher_logits, targets):
        # Direct supervision from the ground-truth pathology targets.
        hard_loss = self.bce(student_logits, targets)


        T = self.temperature


        # Teacher probabilities are fixed targets during student optimisation.
        with torch.no_grad():
            teacher_soft = torch.sigmoid(teacher_logits / T)


        # Compute stable log probabilities for positive and negative output states.
        student_log_prob = F.logsigmoid(student_logits / T)
        student_log_one_minus_prob = F.logsigmoid(-(student_logits / T))


        # Match temperature-scaled teacher and student output probabilities.
        kd_loss = -(
            teacher_soft * student_log_prob
            + (1.0 - teacher_soft) * student_log_one_minus_prob
        ).mean() * (T ** 2)


        # Combine hard-label and logit-distillation terms.
        total = self.alpha_bce * hard_loss + self.alpha_kd * kd_loss
        return total, hard_loss, kd_loss



class FeatureDistillationLoss(nn.Module):
    """Aligns pooled student features with intermediate and final teacher features."""


    def __init__(self, student_dim=192, teacher_mid_dim=384, teacher_final_dim=2208):
        super().__init__()


        # Project student embeddings into the dimensions of teacher feature spaces.
        self.proj_mid = nn.Linear(student_dim, teacher_mid_dim)
        self.proj_final = nn.Linear(student_dim, teacher_final_dim)


    @staticmethod
    def _pool(feat):
        # Mean-pool token sequences; leave non-sequence features unchanged.
        return feat.mean(dim=1) if feat.dim() == 3 else feat


    def forward(self, student_feats, teacher_feats):
        # Pool and project the student's intermediate and final representations.
        s_mid = self.proj_mid(self._pool(student_feats["mid"]))
        s_final = self.proj_final(self._pool(student_feats["final"]))


        # Pool teacher features and detach them to keep the teacher frozen.
        t_mid = self._pool(teacher_feats["mid"]).detach()
        t_final = self._pool(teacher_feats["final"]).detach()


        # Measure alignment at both feature levels.
        loss_mid = F.mse_loss(s_mid, t_mid)
        loss_final = F.mse_loss(s_final, t_final)


        return loss_mid + loss_final