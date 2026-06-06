"""File for concept detection models"""
import torch
import torch.nn as nn
import timm
from torchvision import models


DEFAULT_HEAD_DROPOUT = 0.2


class ConvNeXtModel(nn.Module):
    """ConvNeXt Model"""
    def __init__(self, num_classes, dropout=DEFAULT_HEAD_DROPOUT):
        super().__init__()
        self.model = models.convnext_base(
            weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1
        )

        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.model(x)


class ConvNeXtTinyModel(nn.Module):
    """ConvNeXt-Tiny Model"""
    def __init__(self, num_classes, dropout=DEFAULT_HEAD_DROPOUT):
        super().__init__()
        self.model = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        )

        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.model(x)


class EfficientNetModel(nn.Module):
    """EfficientNet-b0 Model"""
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.efficientnet_b0(weights="IMAGENET1K_V1")

        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class ResNet50Model(nn.Module):
    """ResNet-50 Model"""
    def __init__(self, num_classes, dropout=DEFAULT_HEAD_DROPOUT):
        super().__init__()
        self.model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.model(x)


class TResNetModel(nn.Module):
    """TResNet-M Model"""
    def __init__(
        self,
        num_classes,
        model_name="tresnet_m",
        pretrained=True,
        drop_rate=DEFAULT_HEAD_DROPOUT,
    ):
        super().__init__()

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )

    def forward(self, x):
        return self.model(x)


class EfficientNetV2SmallModel(nn.Module):
    """EfficientNetV2-S Model"""
    def __init__(self, num_classes, dropout=DEFAULT_HEAD_DROPOUT):
        super().__init__()

        weights = (models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.model = models.efficientnet_v2_s(weights=weights)

        in_features = self.model.classifier[1].in_features

        self.model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.model(x)


class DinoV2Model(nn.Module):
    """DINOv2 backbone with a multi-label classification head."""

    def __init__(
        self,
        num_classes,
        backbone_name="dinov2_vitb14",
        freeze_backbone=False,
        dropout=DEFAULT_HEAD_DROPOUT,
        feature_mode: str = "cls_patch_mean",
    ):
        super().__init__()

        if feature_mode not in {"cls", "patch_mean", "cls_patch_mean"}:
            raise ValueError(
                "feature_mode must be one of: 'cls', 'patch_mean', 'cls_patch_mean'"
            )

        self.freeze_backbone = freeze_backbone
        self.feature_mode = feature_mode

        # Needs internet access for first run
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            backbone_name,
            trust_repo=True,
        )

        embed_dim = self.backbone.embed_dim

        if feature_mode == "cls_patch_mean":
            classifier_in_features = embed_dim * 2
        else:
            classifier_in_features = embed_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_in_features),
            nn.Dropout(dropout),
            nn.Linear(classifier_in_features, num_classes),
        )

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

            self.backbone.eval()

    def extract_features(self, x):
        if self.feature_mode == "cls":
            return self.backbone(x)

        outputs = self.backbone.forward_features(x)

        cls_token = outputs["x_norm_clstoken"]
        patch_tokens = outputs["x_norm_patchtokens"]
        patch_mean = patch_tokens.mean(dim=1)

        if self.feature_mode == "patch_mean":
            return patch_mean

        if self.feature_mode == "cls_patch_mean":
            return torch.cat([cls_token, patch_mean], dim=1)

        raise RuntimeError(f"Unexpected feature_mode: {self.feature_mode}")

    def forward(self, x):
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.extract_features(x)
        else:
            features = self.extract_features(x)

        logits = self.classifier(features)

        return logits


class SwinSmallModel(nn.Module):
    """Swin-Small model with a multi-label classification head."""

    def __init__(
        self,
        num_classes,
        pretrained=True,
        freeze_backbone=False,
        dropout=DEFAULT_HEAD_DROPOUT,
    ):
        super().__init__()

        weights = (
            models.Swin_S_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )

        self.model = models.swin_s(weights=weights)

        in_features = self.model.head.in_features

        self.model.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

        if freeze_backbone:
            for name, parameter in self.model.named_parameters():
                if not name.startswith("head."):
                    parameter.requires_grad = False

    def forward(self, x):
        return self.model(x)


def get_model_weights(model_name):
    """Return torchvision weights for models that expose official presets."""
    model_weights = {
        "convnext": models.ConvNeXt_Base_Weights.IMAGENET1K_V1,
        "convnext-tiny": models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
        "efficientnet": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "efficientnetv2": models.EfficientNet_V2_S_Weights.IMAGENET1K_V1,
        "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
        "resnet-50": models.ResNet50_Weights.IMAGENET1K_V2,
        "swinsmall": models.Swin_S_Weights.IMAGENET1K_V1,
    }

    return model_weights.get(model_name)


def load_model(model_name, num_classes, device, **model_kwargs):
    """Load Models"""
    model_dict = {
        "convnext": ConvNeXtModel,
        "convnext-tiny": ConvNeXtTinyModel,
        "efficientnet": EfficientNetModel,
        "efficientnetv2": EfficientNetV2SmallModel,
        "resnet50": ResNet50Model,
        "resnet-50": ResNet50Model,
        "swinsmall": SwinSmallModel,
    }

    if model_name == "dinov2":
        model = DinoV2Model(
            num_classes, backbone_name="dinov2_vitb14", **model_kwargs,
        )
    elif model_name == "tresnet":
        model = TResNetModel(num_classes, model_name="tresnet_m", **model_kwargs)
    else:
        if model_name not in model_dict:
            raise ValueError(f"Unknown model: {model_name}")

        model = model_dict[model_name](num_classes)

    model.to(device)

    return model
