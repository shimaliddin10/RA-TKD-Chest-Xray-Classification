"""
model_teacher.py

ViT-B/16 teacher model for multi-label chest X-ray pathology classification.

Provides a Vision Transformer (ViT-B/16, ImageNet-1K pretrained) backbone
fine-tuned end-to-end for 14 independent thoracic-finding outputs. Supports
optional return of intermediate (mid-encoder) and final token features for
use in feature-level knowledge distillation.

Author: Mohammed Shimal Iddin
Student Number: 40490740
Module: ECS8056: Themed Research Project (2251_SUM)
Supervisor: Dr. Muhammad Fahim
Last Updated: 2026-08-31
"""

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights



class ViTTeacherWithFeatures(nn.Module):
    """ViT-B/16 teacher that optionally exposes intermediate and final token features."""


    def __init__(self, num_classes=14):
        super().__init__()


        # Initialise ViT-B/16 with ImageNet-1K pretrained weights.
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)


        # Replace the original ImageNet head with a target-class prediction layer.
        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_features, num_classes)


    def forward(self, x, return_features=False):
        # Record batch size to expand the learnable class token for each image.
        n = x.shape[0]


        # Convert input images into patch embeddings.
        x = self.model._process_input(x)


        # Prepend the class token and add learned positional embeddings.
        cls_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.model.encoder.pos_embedding


        features = {}


        # Run transformer encoder layers and capture a mid-level token representation.
        for i, layer in enumerate(self.model.encoder.layers):
            x = layer(x)
            if i == 5:
                features["mid"] = x


        # Apply final layer normalisation and retain the final token representation.
        x = self.model.encoder.ln(x)
        features["final"] = x


        # Use the class-token representation for pathology prediction logits.
        logits = self.model.heads(x[:, 0])


        # Return features when feature-level knowledge distillation is required.
        if return_features:
            return {
                "logits": logits,
                "features": features
            }


        return logits