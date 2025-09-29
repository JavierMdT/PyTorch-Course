import torch
from torch import nn
from pathlib import Path
'''
File containing some utility functions for PyTorch model training.
'''

def save_model(model: nn.Module,
               target_dir: str,
               model_name: str) -> None: 
    
    '''
    Saves a PyTorch modelto a target directory.
    
    Args:
        model: Model to save.
        target_dir: Existent or not directory where model will be put in.
        model_name: File name to save the model, must include at the end ".pt" or ".pth".
    
    Example:
        save_model(model=model_0
                   target_dir="models",
                   model_name="TinyVGG.pth")
    '''
    
    # Create target dir if needed
    target_dir_path = Path(target_dir)
    if not target_dir_path.is_dir():
        print(f"Creating '{target_dir}' directory...")
        target_dir_path.mkdir(parents=True)
    else:
        print(f"'{target_dir}' already exists, skipping this step.")
        
    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "Model should end with '.pt' or '.pth'"
    model_save_path = target_dir_path/model_name
    
    # Save the model state_dict
    state_dict = model.state_dict()
    print(f"[INFO]: Saving model to: {str(model_save_path)}")
    torch.save(obj=state_dict,
               f=model_save_path)
    return


def fit_text(text, width):
    text = str(text)
    if len(text) > width:
        return text[:width-3] + "..."
    return text.ljust(width)