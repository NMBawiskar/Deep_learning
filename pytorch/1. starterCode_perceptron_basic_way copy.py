import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

## Get producible results
torch.manual_seed()


class MLP(nn.Module):
    def __init__(self, n_inputs, n_hidden_layers, n_outputs):
        super().__init__()


        ## Add hidden layer
        self.linear1 = nn.Linear(n_inputs, n_hidden_layers)

        self.linear2 = nn.Linear(n_hidden_layers, n_outputs)


    def forward(self, x):
        # Forward pass through hidden layrer
        x = F.relu(self.linear1(x))
        
        return self.linear2(x)



