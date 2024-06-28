import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
def pred_and_plot_image(model: nn.Module,
                         class_names: list,
                         img_path: str,
                         transform: transforms.Compose,
                         device: str = "cpu") -> None:
    """Predicts the class of an image and plots the image with the predicted class."""

    image = Image.open(img_path)
    image_path = Path(img_path)
    image_tensor = transform(image).unsqueeze(0).to(device) # unsqueeze to add batch dimension
    model.to(device)
    model.eval()
    with torch.inference_mode():
        preds = model(image_tensor)
        label = class_names[torch.argmax(preds, dim = 1).item()]
        plt.figure(figsize=(10,8))
        plt.subplot(1,2,1)
        plt.imshow(image)
        plt.title(f"Actual: {image_path.parent.stem}")
        plt.axis("off")
        plt.subplot(1,2,2)
        image_tensor = image_tensor[0].to('cpu').detach().numpy()
        image_tensor = np.transpose(image_tensor, (1,2,0))
        image_tensor = np.clip(image_tensor, 0, 1)
        plt.imshow(image_tensor)
        plt.title(f"Prediction: {label}")
        plt.axis("off")
        plt.show()