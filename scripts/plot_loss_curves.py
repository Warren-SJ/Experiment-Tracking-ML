import matplotlib.pyplot as plt
from typing import Dict, List

def plot_loss_curves(
        history: Dict[str, List[float]]) -> None:
    """Plots the training loss, training accuracy, validation loss and validation accuracy.
    Args:
        history: a dictionary containing the training history.
    Returns:
        None
    """
    num_epochs = len(history["train_loss"])
    plt.figure(figsize=(10, 7))
    plt.subplot(1,2,1)
    plt.plot(range(num_epochs), history["train_loss"], label="train_loss")
    plt.plot(range(num_epochs), history["val_loss"], label="val_loss")
    plt.title("Loss vs. epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(range(num_epochs), history["train_acc"], label="train_acc")
    plt.plot(range(num_epochs), history["val_acc"], label="val_acc")
    plt.title("Accuracy vs. epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()