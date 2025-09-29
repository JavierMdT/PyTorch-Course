import torch
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List
from torch import nn

def train_step(model: nn.Module,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               dataloader: DataLoader,
               dev: str) -> Tuple[float, float]:

    '''
    Performs the train step. This function should be called per epoch.
    
    Args:
        model: PyTorch model to train.
        loss_fn: Loss function to minimize.
        optimizer: Optimizer that will make the model learn.
        dataloader: The train dataloader to train the model using batches.
        dev: Device where hard-computation will be performed.
        
    Return:
        This function returns a tuple that gives: [train_loss, train_accuracy].
    '''
    
    
    train_loss, train_acc = 0, 0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(dev), y.to(dev)
        
        # Prediction
        logits = model(X)
        pred = torch.argmax(logits, dim=1)
        
        # Metrics
        batch_loss = loss_fn(logits, y)
        train_loss += batch_loss.item()
        train_acc += (y==pred).sum().item()/(len(y))
        
        # Algorithm steps ç
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    
    return (train_loss, train_acc)

def test_step(model: nn.Module,
              loss_fn: nn.Module,
              dataloader: DataLoader,
              dev: str) -> Tuple[float, float]:

    '''
    Performs the train step. This function should be called per epoch.
    
    Args:
        model: PyTorch model to train.
        loss_fn: loss function to evaluate the model.
        dataloader: The test dataloader to test the model using batches.
        dev: Device where hard-computation will be performed.
        
    Return:
        This function returns a tuple that gives: [test_loss, test_accuracy].
    '''
    
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        model.eval()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(dev), y.to(dev)
            
            logits = model(X)
            pred = torch.argmax(logits, dim=1)
            
            batch_loss = loss_fn(logits, y).item()
            test_loss += batch_loss
            test_acc += (y==pred).sum().item()/(len(y))
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)

    return (test_loss, test_acc)

def train(model: nn.Module,
          loss_fn: nn.Module,
          optimizer: torch.optim.Optimizer,
          train_dataloader: DataLoader,
          test_dataloader: DataLoader,
          epochs: int,
          dev: str) -> Dict[str, List[float]]: 
    '''
    This function trains and tests the model performance per epoch.
    Even though it prints out the results for each epoch, it also saves them in a dict.
    
    Args:
        model: Model that it's gonna be trained.
        loss_fn: Loss function to minimize.
        optimizer: Optimizer that it's gonna improve the model.
        train_dataloader: Dataloder for train step.
        test_dataloader: Dataloader for test step.
        epochs: Number of epochs to train the model.
        dev: Device where hard-computation it's gonna happen.
        
    Returns: 
        This functions returns a dict of 4 metrics in this format: 
            "train_loss": []
            "train_acc":  []
            "test_loss":  []
            "test_acc":   []
        For every key, the lenght of the list will be equal to the nº of epochs. 
        
    Example:
        For 4 epochs: 
        Results = {
            "train_loss": [..., ..., ..., ...]
            "train_acc":  [..., ..., ..., ...]
            "test_loss":  [..., ..., ..., ...]
            "test_acc":   [..., ..., ..., ...]
        }
    '''
    
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    for e in range(epochs):
        print(f"-------------- Epoch {e} --------------")
        train_loss: float
        train_acc: float
        train_loss, train_acc = train_step(model=model,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           dataloader=train_dataloader,
                                           dev=dev)
        
        test_loss: float
        test_acc: float
        test_loss, test_acc = test_step(model=model,
                                           loss_fn=loss_fn,
                                           dataloader=test_dataloader,
                                           dev=dev)
        
        print(f"Train metrics: ({train_loss:.4f}, {train_acc:.2f})\nTest metrics: ({test_loss:.4f}, {test_acc:.2f})")
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    print("-------------------------------------")
    
    return results 
