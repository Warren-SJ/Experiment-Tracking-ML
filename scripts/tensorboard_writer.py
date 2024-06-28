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