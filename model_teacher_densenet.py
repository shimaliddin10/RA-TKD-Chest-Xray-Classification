"""
model_teacher_densenet.py

DenseNet161 teacher model for multi-label chest X-ray pathology classification.

Provides a DenseNet161 backbone (ImageNet-1K pretrained) fine-tuned end-to-end
for 14 independent thoracic-finding outputs. Supports optional return of
intermediate (384-channel) and final (2208-channel) pooled features for use
in feature-level knowledge distillation.

Author: Mohammed Shimal Iddin
Student Number: 40490740
Module: ECS8056: Themed Research Project (2251_SUM)
Supervisor: Dr. Muhammad Fahim
Last Updated: 2026-08-31
"""

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet161, DenseNet161_Weights



class DenseNet161TeacherWithFeatures(nn.Module):
    """DenseNet161 teacher that optionally returns intermediate and final features."""


    def __init__(self, num_classes=14):
        super().__init__()


        # Initialise DenseNet161 with ImageNet-1K pretrained weights.
        self.model = densenet161(weights=DenseNet161_Weights.IMAGENET1K_V1)


        # Replace the original classifier with an output layer for the target classes.
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)


    def forward(self, x, return_features=False):
        # Access DenseNet feature-extraction layers to expose distillation features.
        feats = self.model.features


        # Initial convolutional feature extraction.
        x = feats.conv0(x)
        x = feats.norm0(x)
        x = feats.relu0(x)
        x = feats.pool0(x)


        # Early and intermediate DenseNet blocks.
        x = feats.denseblock1(x)
        x = feats.transition1(x)
        x = feats.denseblock2(x)
        x = feats.transition2(x)
        mid_map = x  # 384-channel intermediate feature map.


        # Later DenseNet blocks for final high-level feature extraction.
        x = feats.denseblock3(x)
        x = feats.transition3(x)
        x = feats.denseblock4(x)
        final_map = feats.norm5(x)  # 2208-channel final feature map.


        # Spatially pool feature maps to create vectors for feature distillation.
        mid_pooled = F.adaptive_avg_pool2d(F.relu(mid_map), (1, 1)).flatten(1)
        final_pooled = F.adaptive_avg_pool2d(F.relu(final_map), (1, 1)).flatten(1)


        # Produce pathology prediction logits from final pooled features.
        logits = self.model.classifier(final_pooled)


        # Return representations only when feature-level distillation is required.
        if return_features:
            return {
                "logits": logits,
                "features": {
                    "mid": mid_pooled,
                    "final": final_pooled
                }
            }


        return logits