
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from typing import List, Tuple
import os 

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers:int=NUM_WORKERS
    ) -> Tuple[DataLoader, DataLoader, List[str]]:
    
    '''Creates train & test DataLoaders.
    
    Takes in a training directory & testing directory and turns them int PyTorch 
    Datasets and then into PyTorch DataLoaders.
    
    Args: 
        train_dir = Path to training directory.
        test_dir = Same for test directory.
        transform = torchvision tranforms to perform on train & test data.
        batch_size = Number of samples per batch in each of the DataLoaders.
        num_workers = An int for a number of workers per DataLoader.
        
    Returns: 
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
        Example usage:
            train_dataloader, test_dataloader, class_names = create_dataloaders(
                train_dir=train_path,
                test_dir=test_path,
                transform=some_transform,
                batch_size=32,
                num_workers=4)
    '''

    # Use ImageFolder to create the Dataset(s)
    train_data = datasets.ImageFolder(root=str(train_dir),
                                    transform=transform,
                                    target_transform=None)
    test_data = datasets.ImageFolder(root=str(test_dir),
                                    transform=transform,
                                    target_transform=None)

    # Get classes
    class_names = train_data.classes

    # Print out results
    print(f"Train data:\n{train_data}\nTest data:\n{test_data}")

    # Get DataLoader(s)
    train_dataloader = DataLoader(dataset=train_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=True)
    test_dataloader = DataLoader(dataset=test_data,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True)

    
    
    return train_dataloader, test_dataloader, class_names
