import torch
from model import EfficientNetb2
from pathlib import Path
import gradio as gr
import time
from typing import Tuple, Dict
class_names = ["pizza", "steak", "sushi"]
file_dir = Path(__file__).resolve().parent
test_images = Path(file_dir/"images").rglob("*.jpg")

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
    demo.launch(share = False)

if __name__ == "__main__":
    main()