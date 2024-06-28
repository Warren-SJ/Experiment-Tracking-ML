from pathlib import Path
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

NUM_WORKERS = 0

def create_dataloaders(
    data_dir: str,
    batch_size: int,
    train_transform: transforms.Compose,
    test_transform: transforms.Compose = None,
    num_workers: int = NUM_WORKERS,
    ):
    """Creates DataLoader objects for the train and test datasets
    Args:
        data_dir: str: path to the data directory. This directory should contain two subdirectories, 'train' and 'test'
                       Within each subdirectory should be a subdirectory which contins relevant images for each class
        batch_size: int: number of samples per batch
        train_transform: transforms.Compose: a series of preprocessing transformations on the train data
        test_transform: transforms.Compose: a series of preprocessing transformations on the test data. If not provided, the train_transform will be used
        num_workers: int: number of subprocesses to use for data loading
        
        Returns:
            train_dataloader: DataLoader: a DataLoader object for the train dataset
            test_dataloader: DataLoader: a DataLoader object for the test dataset
            class_names: list: a list of class names"""
    
    # Define data directories
    train_dir = Path(data_dir) / 'train'
    test_dir = Path(data_dir) / 'test'

    # Create datasets
    if test_transform is None:
        test_transform = train_transform

    train_dataset = ImageFolder(root = train_dir,
                                transform = train_transform,
                                target_transform=None)
    
    test_dataset = ImageFolder(root = test_dir,
                                 transform = test_transform,
                                 target_transform=None)
    class_names = train_dataset.classes

    # Create dataloaders
    train_dataloader = DataLoader(dataset = train_dataset,
                                  batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = num_workers)
    test_dataloader = DataLoader(dataset = test_dataset,
                                    batch_size = batch_size,
                                    shuffle = False,
                                    num_workers = num_workers)
    
    return train_dataloader, test_dataloader, class_names