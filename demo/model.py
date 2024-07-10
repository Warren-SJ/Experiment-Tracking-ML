import torch
from torch import nn
from torchvision import models
from pathlib import Path


class EfficientNetb2(nn.Module):
    def __init__(self, num_classes: int, state_dict_path: Path):
        super(EfficientNetb2, self).__init__()
        self.model = models.efficientnet_b2(weights=None)
        self.transforms = models.EfficientNet_B2_Weights.DEFAULT.transforms()
        self.model.classifier = nn.Sequential(nn.Dropout(p=0.3, inplace=True),
                                              nn.Linear(in_features=1408, out_features=num_classes, bias=True))
        self.model.load_state_dict(torch.load(state_dict_path))
    def forward(self, x):
        return self.model(x)