"""Load a pretrained ResNet18 and replace its last layer."""
from torch import nn
from torchvision import models


def build_model(num_classes=2, freeze=False, pretrained=True):
    weights = None
    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters()
                if p.requires_grad)
    return total, train


if __name__ == "__main__":
    for freeze in (False, True):
        total, train = count_params(build_model(freeze=freeze))
        print(f"freeze={freeze}: {train:,} of {total:,}"
              " parameters trainable")
