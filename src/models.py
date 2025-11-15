"""Definición de modelos utilizados en inferencia."""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class MultimodalRegressor(nn.Module):
    """Modelo que fusiona características de texto e imagen."""

    def __init__(self, text_dim: int, dropout: float = 0.3, text_hidden: int = 256, fusion_hidden: int = 256):
        super().__init__()
        self.image_backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        image_feat_dim = self.image_backbone.fc.in_features
        self.image_backbone.fc = nn.Identity()

        self.text_mlp = nn.Sequential(
            nn.Linear(text_dim, text_hidden),
            nn.LayerNorm(text_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.regressor = nn.Sequential(
            nn.Linear(image_feat_dim + text_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(self, images, text_feat):
        img_feat = self.image_backbone(images)
        txt_feat = self.text_mlp(text_feat)
        fused = torch.cat([img_feat, txt_feat], dim=1)
        return self.regressor(fused)
