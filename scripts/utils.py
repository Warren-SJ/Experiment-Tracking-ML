import torch
from torch import nn
from pathlib import Path

def save_model(model:nn.Module,
               target_dir: str,
               model_name: str):
    """Saves the model's state_dict to a file in the path specified.
    Saves as /target_dir/model_name.pth
    Args:
        model: The neural network model to be saved.
        model_path: The path where the model will be saved.
    Returns:
        None"""
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    model_name = model_name + '.pth'
    model_path = target_dir_path / model_name
    print(f"Saving model to {model_path}")
    torch.save(obj = model.state_dict(), 
               f = model_path)