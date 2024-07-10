from torchvision import models
from torchvision import transforms
import torch
from torch import nn
from pathlib import Path

class EfficientNetb2(nn.Module):
    def __init__(self, num_classes: int, state_dict_path: Path):
        super(EfficientNetb2, self).__init__()
        self.model = models.efficientnet_b2(weights=None)
        self.transforms = models.EfficientNet_B2_Weights.DEFAULT.transforms()
        self.model.classifier = nn.Sequential(nn.Dropout(p=0.3, inplace=True),
                                              nn.Linear(in_features=1408, out_features=num_classes, bias=True))
        self.model.load_state_dict(torch.load(state_dict_path))
        # self.transforms = transforms.Compose([
        #         transforms.Resize(288, interpolation=transforms.InterpolationMode.BICUBIC),
        #         transforms.CenterCrop(288),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])
    def forward(self, x):
        return self.model(x)
    
