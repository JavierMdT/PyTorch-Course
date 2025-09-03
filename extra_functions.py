import matplotlib.pyplot as plt 
import torch

def print_progress(progress: int):
    '''
    A simple function to print the progress of a loop
    The progress argument must be between 0 and 10
    It's a 10 parts progress bar
    Args:
        progress(int) = done(int) // all(int) 
    '''
    left: int = 10 - progress
    bar: str = progress*"⬜" + left*"⬛"
    print(f"\rTrain progress: {bar}", end="", flush=True)
    
    
    
def plot_pred_images(preds:list,
                     y: list,
                     class_names: list,
                     figsize: tuple, 
                     nrows: int,
                     ncols: int,
                     images: list) -> None:
    
    fig = plt.figure(figsize=figsize)
    
    for idx in range(5):
        fig.add_subplot(nrows, ncols, idx+1)
        plt.imshow(images[idx].squeeze())
        if preds[idx] == y[idx]:
            plt.title(class_names[y[idx]], c="g")
        else:
            plt.title(class_names[y[idx]], c="r")