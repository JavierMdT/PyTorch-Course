import torch
from torch import nn

'''
Contains PyTorch model code to instantiate a TinyVGG model from the CNN Explainer website.
'''

class TinyVGG(nn.Module):
    '''Creates the TinyVGG architecture.
    
    Args:
        in_c: An integer indicating number of input channels.
        out_shape: An integer indicating number of classes.
        hidden_units: An integer indicating number of hidden units between layers. 
    '''
    def __init__(self,
                 in_c:int,
                 out_shape:int,
                 hidden_units:int) -> None:
        super().__init__()
        
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_c,
                      out_channels=hidden_units,
                      stride=1,
                      padding=0,
                      kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      stride=1,
                      padding=0,
                      kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),
                         stride=2)
        )
        
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      stride=1,
                      padding=0,
                      kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      stride=1,
                      padding=0,
                      kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),
                         stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 13 * 13,
                      out_features=out_shape)
        )
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:    
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))
