import torch
from torchvision import transforms
from torchvision import models
from torch import nn
from pathlib import Path
import gradio as gr
import time
import random
from typing import Tuple, Dict
class_names = ["pizza", "steak", "sushi"]
file_dir = Path(__file__).resolve().parent
test_images = Path(file_dir/"images").rglob("*.jpg")

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
model = EfficientNetb2(num_classes=len(class_names), state_dict_path="models/efficientnet_b2_new.pth").to("cpu")


def predict(img) -> Tuple[Dict, float]: 
    """Takes an image as an input and returns the predicted class and the probability of the prediction"""
    start_time = time.time()
    img = model.transforms(img).unsqueeze(0)
    model.eval()
    with torch.inference_mode():
        pred_logits = model(img)
        pred_probs = torch.softmax(pred_logits, dim=1)
    pred_probs_and_labels = {class_names[i]: pred_probs[0][i] for i in range(len(class_names))}
    end_time = time.time()
    return pred_probs_and_labels, end_time - start_time

def main():
    example_list = [str(file_path) for file_path in test_images]
    print(example_list)
    demo = gr.Interface(fn = predict,
                    inputs =gr.Image(type="pil"),
                    outputs =[gr.Label(num_top_classes=3, label="Predictions"), gr.Number(label="Prediction time (s)")],
                    examples = example_list,
                    title = "EfficientNet B2 Image Classifier",
                    description="This is an image classifier which uses an EfficientNet B2 model to classify images of pizza, steak and sushi"
                    )
    demo.launch()

if __name__ == "__main__":
    main()