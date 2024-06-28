import torch
from torch import nn

class TinyVGG(nn.Module):
    """Replicates the TinyVGG architecture as stated in the CNN Explainer website.
    Args:
        input_shape = number of channels in the  image. 3 for RGB, 1 for grayscale.
        hidden_units = number of hidden units in the fully connected layer.
        num_classes = number of classes in the dataset."""
    
    def __init__(self, input_shape:int,hidden_units:int, num_classes:int):
        super().__init__()
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(in_channels = input_shape, 
                      out_channels = hidden_units, 
                      kernel_size = 3, 
                      stride = 1, 
                      padding = 0),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_units, 
                      out_channels = hidden_units, 
                      kernel_size = 3, 
                      stride = 1, 
                      padding = 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, 
                         stride = 2)
        )
        self.convBlock2 = nn.Sequential(
            nn.Conv2d(in_channels = hidden_units, 
                      out_channels = hidden_units, 
                      kernel_size = 3, 
                      stride = 1, 
                      padding = 0),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_units, 
                      out_channels = hidden_units, 
                      kernel_size = 3, 
                      stride = 1, 
                      padding = 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, 
                         stride = 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = 13 * 13 * hidden_units,
                      out_features = num_classes),
        )
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        return self.classifier(self.convBlock2(self.convBlock1(X)))
        