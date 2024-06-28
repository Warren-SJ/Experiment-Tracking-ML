import torch
import data_setup, utils, engine, model_setup, data_acquisition
import sys
from torchvision import transforms

try:
    MODEL_NAME = sys.argv[1]
    BATCH_SIZE = int(sys.argv[2])
    LEARNING_RATE = float(sys.argv[3])
    NUM_EPOCHS = int(sys.argv[4])
except IndexError:
    MODEL_NAME = "tiny_vgg"
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 5
    
NUM_WORKERS = 0

data_dir = "data/pizza_steak_sushi"

data_acquisition.acquire_data(image_path = "pizza_steak_sushi",
                              url = "https://github.com/mrdbourke/pytorch-deep-learning/blob/0fa794be523a10b409a2061e43ae03c419d5ace7/data/pizza_steak_sushi_20_percent.zip?raw=true",
                              zip_name = "pizza_steak_sushi.zip",)

device = "cuda" if torch.cuda.is_available() else "cpu"

train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.TrivialAugmentWide(num_magnitude_bins=5),
    transforms.ToTensor()
])
val_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_dataloader, val_dataloader, class_names = data_setup.create_dataloaders(data_dir=data_dir,
                                                                               batch_size = BATCH_SIZE,
                                                                                train_transform=train_transform,
                                                                                test_transform=val_transform,
                                                                                num_workers=NUM_WORKERS)

model = model_setup.TinyVGG(input_shape = 3,
                            hidden_units = 10,
                            num_classes = len(class_names)).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

engine.train(model = model,
             model_name = MODEL_NAME,
             data_name = "pizza_steak_sushi",
             epochs = NUM_EPOCHS,
             train_dataloader=train_dataloader,
             val_dataloader=val_dataloader,
             device=device,
             loss_fn=loss_fn,
             optimizer=optimizer)

utils.save_model(model = model,
                 target_dir = "models",
                 model_name = MODEL_NAME)

print("Model training complete!")