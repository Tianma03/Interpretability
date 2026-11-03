"""Explicit ResNet18 backbone for concept bottleneck models.

ResNet architecture follows torchvision's ResNet implementation:
https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
Consult torchvision's BSD-3-Clause license when redistributing adapted code.
"""
from typing import Callable, Optional

import torch
from torch import Tensor
from torch import nn
from torchvision.models import ResNet18_Weights


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 is not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class ResNet(nn.Module):
    """ResNet with explicit layers and an accessible feature extractor.

    While fc is a Linear layer, forward(x) returns class logits.
    Replacing fc with nn.Identity() makes forward(x) return pooled features.
    """

    def __init__(
        self,
        block: type[BasicBlock],
        layers: list[int],
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 512 * block.expansion
        self.fc = nn.Linear(self.feature_dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self, block: type[BasicBlock], planes: int, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        if blocks < 1:
            raise ValueError("blocks must be >= 1")
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                self._norm_layer(planes * block.expansion),
            )
        stage = [
            block(
                self.inplanes, planes, stride=stride, downsample=downsample,
                groups=self.groups, base_width=self.base_width,
                dilation=self.dilation, norm_layer=self._norm_layer,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            stage.append(
                block(
                    self.inplanes, planes,
                    groups=self.groups, base_width=self.base_width,
                    dilation=self.dilation, norm_layer=self._norm_layer,
                )
            )
        return nn.Sequential(*stage)

    def forward_features(self, x: Tensor, return_maps: bool = False):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        features = torch.flatten(self.avgpool(f4), 1)
        if return_maps:
            return features, {"layer1": f1, "layer2": f2, "layer3": f3, "layer4": f4}
        return features

    def forward(self, x: Tensor) -> Tensor:
        features = self.forward_features(x)
        return self.fc(features)

class ResNet18PTCVCUB(nn.Module):
    """
    ResNet18 pretrained on CUB-200-2011 using pytorchcv.

    features_only=True:
        return [B, 512] features

    features_only=False:
        return [B, 200] class logits
    """

    feature_dim = 512

    def __init__(
        self,
        pretrained=True,
        features_only=True,
    ):
        super().__init__()

        from pytorchcv.model_provider import get_model

        base = get_model(
            "resnet18_cub",
            pretrained=pretrained,
        )

        # PTCV ResNet 
        self.init_block = base.features.init_block
        self.layer1 = base.features.stage1
        self.layer2 = base.features.stage2
        self.layer3 = base.features.stage3
        self.layer4 = base.features.stage4
        self.avgpool = base.features.final_pool

        # 
        self.fc = (
            nn.Identity()
            if features_only
            else base.output
        )

    def forward_features(self, x, return_maps=False):

        x = self.init_block(x)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        features = self.avgpool(f4)
        features = torch.flatten(features, 1)

        if return_maps:
            maps = {
                "layer1": f1,
                "layer2": f2,
                "layer3": f3,
                "layer4": f4,
            }
            return features, maps

        return features

    def forward(self, x):
        features = self.forward_features(x)
        return self.fc(features)


def resnet18_ptcv_cub(pretrained=True,features_only=True,):
    return ResNet18PTCVCUB(pretrained=pretrained,features_only=features_only,)

class TorchVisionViT(nn.Module):
    """ViT-B/16 image encoder, exposing pooled CLS and spatial patch outputs."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import ViT_B_16_Weights, vit_b_16 as tv_vit_b_16

        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vit = tv_vit_b_16(weights=weights)
        self.feature_dim = self.vit.hidden_dim
        self.patch_size = self.vit.patch_size
        self.image_size = self.vit.image_size
        # The classifier is not used in CBM feature extraction.
        self.vit.heads = nn.Identity()

    def forward_features(self, x: Tensor, return_maps: bool = False):
        # Official VisionTransformer patchification and encoder sequence.
        patch_tokens = self.vit._process_input(x)
        batch = patch_tokens.shape[0]
        cls = self.vit.class_token.expand(batch, -1, -1)
        tokens = self.vit.encoder(torch.cat([cls, patch_tokens], dim=1))
        cls_feature = tokens[:, 0]                   # [B, 768]

        if not return_maps:
            return cls_feature

        patch_tokens = tokens[:, 1:]                 # [B, 196, 768]
        height, width = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        patch_map = patch_tokens.transpose(1, 2).reshape(
            batch, self.feature_dim, height, width
        )
        return cls_feature, {"patch_tokens": patch_tokens, "patch_map": patch_map}

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_features(x)
        
def vit_b_16(pretrained: bool = True):
    return TorchVisionViT(pretrained=pretrained)   
   
class CLIPViT(nn.Module):
    """CLIP's VISUAL encoder only, returning pre-projection CLS representation.

    Not the CLIP image-text aligned projection! The raw hidden dimension for
    openai/clip-vit-base-patch16 is 768, rather than the projected 512.
    """

    def __init__(
        self,
        pretrained: bool = True,
        model_name: str = "openai/clip-vit-base-patch16",
    ):
        super().__init__()
        try:
            from transformers import CLIPVisionConfig, CLIPVisionModel
        except ImportError as exc:
            raise ImportError("CLIPViT needs transformers: pip install transformers") from exc

        if pretrained:
            self.vision = CLIPVisionModel.from_pretrained(model_name)
        else:
            # Offline/random initialization for the named ViT-B/16 architecture.
            if model_name != "openai/clip-vit-base-patch16":
                raise ValueError("pretrained=False currently supports only CLIP ViT-B/16")
            cfg = CLIPVisionConfig(
                image_size=224,
                patch_size=16,
                hidden_size=768,
                intermediate_size=3072,
                num_hidden_layers=12,
                num_attention_heads=12,
            )
            self.vision = CLIPVisionModel(cfg)

        self.feature_dim = self.vision.config.hidden_size
        self.patch_size = self.vision.config.patch_size
        self.image_size = self.vision.config.image_size

    def forward_features(self, x: Tensor, return_maps: bool = False):
        output = self.vision(pixel_values=x, return_dict=True)
        cls_feature = output.pooler_output          # [B, 768], post LayerNorm CLS
        if not return_maps:
            return cls_feature

        patch_tokens = output.last_hidden_state[:, 1:, :]  # [B, 196, 768]
        batch = x.shape[0]
        height, width = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        if patch_tokens.shape[1] != height * width:
            raise ValueError("Patch token count mismatch; check image size / position embeddings")
        patch_map = patch_tokens.transpose(1, 2).reshape(
            batch, self.feature_dim, height, width
        )
        return cls_feature, {"patch_tokens": patch_tokens, "patch_map": patch_map}

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_features(x)


def clip_vit_b_16(pretrained: bool = True):
    return CLIPViT(pretrained=pretrained)     
        
           
def resnet18(pretrained: bool = False, features_only: bool = True) -> ResNet:
    """Make ResNet18; optionally load official ImageNet weights.

    features_only=True: returns [B,512] rather than ImageNet class logits.
    When pretrained=True, network download may be required the first time.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1000)
    if pretrained:
        weights = ResNet18_Weights.DEFAULT.get_state_dict(progress=True, check_hash=True)
        model.load_state_dict(weights, strict=True)
    if features_only:
        model.fc = nn.Identity()
    return model


def get_backbone(name: str, pretrained: bool = True):
    """Return a feature-only backbone. Feature dimension: model.feature_dim."""
    constructors = {
        "resnet18": lambda: resnet18(pretrained=pretrained, features_only=True),
        "resnet18_ptcv_cub": lambda: resnet18_ptcv_cub(pretrained=pretrained, features_only=True),
        "vit_b_16": lambda: vit_b_16(pretrained=pretrained),
        "clip_vit_b_16": lambda: clip_vit_b_16(pretrained=pretrained),
    }
    if name not in constructors:
        raise ValueError(f"Unknown backbone {name!r}; choose from {list(constructors)}")
    return constructors[name]()
