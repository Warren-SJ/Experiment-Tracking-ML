import torch
from torch import nn
from tqdm.auto import tqdm
from typing import List, Tuple, Dict
from torch.utils.tensorboard.writer import SummaryWriter
import os
import datetime

def create_writer(experiment_name : str,
                  model_name : str,
                  epochs : int):
    """Creates an instance of tensorflow's SummaryWriter class. It saves the logs in the directory
    `runs/[date][experiment_name][model_name][epochs]`.
    Args:
        experiment_name (str): Name of the experiment. Preferebly the name of the dataset.
        model_name (str): Name of the model.
        epochs (int): Number of epochs.
    Returns:
        SummaryWriter: Instance of SummaryWriter class.
    """
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("runs", f"{current_time}_{experiment_name}_{model_name}_{epochs}")
    writer = SummaryWriter(log_dir=log_dir)
    return writer

def train_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               device: torch.device = torch.device('cpu'),
               loss_fn: nn.Module = nn.CrossEntropyLoss(),
               optimizer: torch.optim.Optimizer = torch.optim.Adam,) -> Tuple[float, float]:
    """Trains the model for one epoch.
    Args:
        model: The neural network model used for the classifier.
        dataloader: The torch dataloader object for training.
        loss_fn: The loss function to be used. Default is CrossEntropyLoss.
        optimizer: The optimizer to be used. Default is Adam.
        device: The device to run the training on. Default is CPU.
    Returns:
        A tuple of two floats: the training loss and accuracy of the model."""
    model.train()
    train_loss, train_acc = 0.0, 0.0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        y_preds = model(X)
        loss = loss_fn(y_preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += ((torch.argmax(y_preds, dim = 1) == y).sum().item() / len(y))
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc


def eval_step(model: nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: nn.Module = nn.CrossEntropyLoss(),
              device: torch.device = torch.device('cpu')) -> Tuple[float, float]:
    """Evaluates the model on the validation set.
    Args:
        model: The neural network model used for the classifier.
        dataloader: The torch dataloader object for validation.
        loss_fn: The loss function to be used. Default is CrossEntropyLoss.
        device: The device to run the evaluation on. Default is CPU.
    Returns:
        A tuple of two floats: the validation loss and accuracy of the model,
        predictions for all validation samples
        true labels for all validation samples."""
    model.eval()
    all_preds, all_labels = [], []
    val_loss, val_acc = 0.0, 0.0
    with torch.inference_mode():
        for X,y in dataloader:
            X, y = X.to(device), y.to(device)
            y_preds = model(X)
            loss = loss_fn(y_preds, y)
            val_loss += loss.item()
            val_acc += ((torch.argmax(y_preds, dim = 1)) == y).sum().item() / len(y)
            all_preds.append(torch.argmax(y_preds, dim = 1))
            all_labels.append(y)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    val_loss /= len(dataloader)
    val_acc /= len(dataloader)

    return val_loss, val_acc, all_preds, all_labels

def train(model: nn.Module,
          model_name: str,
          data_name: str,
          epochs: int,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          device: torch.device = torch.device('cpu'),
          loss_fn: nn.Module = nn.CrossEntropyLoss(),
          optimizer: torch.optim.Optimizer = torch.optim.Adam) -> Dict[str, List[float]]:
    """Trains the model for the specified number of epochs and returns the training history
    including training and validation loss and accuracy.
    Args:
        model: The neural network model used for the classifier.
        model_name: The name of the model.
        data_name: The name of the dataset. eg: MNIST, CIFAR-10
        epochs: The number of epochs to train the model for.
        train_dataloader: The torch dataloader object for training.
        val_dataloader: The torch dataloader object for validation.
        device: The device to run the training on. Default is CPU.
        loss_fn: The loss function to be used. Default is CrossEntropyLoss.
        optimizer: The optimizer to be used. Default is Adam.
    Returns:
        A dictionary containing the training history.
        Predictions for all validation samples.
        True labels for all validation samples.
        A SummaryWriter object for logging the training process.
        """
    writer = create_writer(experiment_name = data_name,
                           model_name = model_name,
                           epochs = epochs)
    history = {"train_loss" : [],
               "train_acc" : [],
               "val_loss" : [],
               "val_acc" : []}
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model = model,
            dataloader = train_dataloader,
            device = device,
            loss_fn = loss_fn,
            optimizer = optimizer
        )
        val_loss, val_acc, all_preds, all_labels = eval_step(
                                        model = model,
                                        dataloader = val_dataloader,
                                        loss_fn = loss_fn,
                                        device = device
        )
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        writer.add_scalar(tag = "Loss/Train", scalar_value = train_loss, global_step = epoch)
        writer.add_scalar(tag = "Loss/Val", scalar_value = val_loss, global_step = epoch)
        writer.add_scalar(tag = "Accuracy/Train", scalar_value = train_acc, global_step = epoch)
        writer.add_scalar(tag = "Accuracy/Val", scalar_value = val_acc, global_step = epoch)
        print(f"Epoch {epoch+1} of {epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return history, all_preds, all_labels, writer

def create_writer(experiment_name : str,
                  model_name : str,
                  epochs : int):
    """Creates an instance of tensorflow's SummaryWriter class. It saves the logs in the directory
    `runs/[date][experiment_name][model_name][epochs]`.
    Args:
        experiment_name (str): Name of the experiment. Preferebly the name of the dataset.
        model_name (str): Name of the model.
        epochs (int): Number of epochs.
    Returns:
        SummaryWriter: Instance of SummaryWriter class.
    """
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("runs", f"{current_time}_{experiment_name}_{model_name}_{epochs}")
    writer = SummaryWriter(log_dir=log_dir)
    return writer