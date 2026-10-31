"""ResNet50 backbone wrapper: fine-tune or frozen feature-extractor mode."""
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

FEATURE_DIM = 2048


class ResNetBackbone(nn.Module):
    """ResNet50 with its final fc replaced; exposes both features and logits.

    mode='finetune': all weights trainable (B1-tuned, B3-A).
    mode='frozen': backbone weights frozen, only used as a feature extractor (B2).
    """

    def __init__(self, num_classes, mode="finetune", pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = resnet50(weights=weights)
        self.features = nn.Sequential(*list(net.children())[:-1])  # drop original fc
        self.fc = nn.Linear(FEATURE_DIM, num_classes)
        self.mode = mode
        if mode == "frozen":
            for p in self.features.parameters():
                p.requires_grad = False

    def extract_features(self, x):
        feat = self.features(x)
        return feat.flatten(1)  # (B, 2048)

    def forward(self, x):
        feat = self.extract_features(x)
        return self.fc(feat)

    def train(self, mode=True):
        super().train(mode)
        if self.mode == "frozen":
            self.features.eval()  # keep BN stats fixed even if caller calls .train()
        return self
