
import torch
from torch import nn

from .backbone import get_backbone


class JointCBM(nn.Module):
    def __init__(
        self,
        backbone_name="resnet18_ptcv_cub",
        pretrained=True,
        num_concepts=112,
        num_classes=200,
        concept_activation="none",
        freeze_backbone=False,
    ):
        super().__init__()

        if concept_activation not in ("none", "sigmoid"):
            raise ValueError("Invalid concept activation")

        self.concept_activation = concept_activation

        self.backbone = get_backbone(
            backbone_name,
            pretrained=pretrained
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

        self.dropout = nn.Dropout(p=0.5)

        self.concept_heads = nn.ModuleList([
            nn.Linear(self.backbone.feature_dim, 1)
            for _ in range(num_concepts)
        ])

        for head in self.concept_heads:
            nn.init.trunc_normal_(
                head.weight, std=0.1, a=-0.2, b=0.2
            )

        self.classifier = nn.Linear(
            num_concepts, num_classes
        )

    def forward(self, images):

        features = self.dropout(
            self.backbone(images)
        )

        concept_logits = torch.cat(
            [head(features) for head in self.concept_heads],
            dim=1
        )

        concepts = (
            concept_logits.sigmoid()
            if self.concept_activation == "sigmoid"
            else concept_logits
        )

        class_logits = self.classifier(concepts)

        return [
            class_logits,
            *concept_logits.split(1, dim=1)
        ]
