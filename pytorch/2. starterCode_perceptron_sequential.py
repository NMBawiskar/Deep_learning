import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

## Get producible results
torch.manual_seed(42)


class MLP(nn.Module):
    def __init__(self, n_inputs, n_hidden_layers, n_outputs):
        super().__init__()


        self.model = nn.Sequential(
            ##
            nn.Linear(n_inputs, n_hidden_layers),

            ## ADD ReLU actiavation
            nn.ReLU(),

            nn.Linear(n_hidden_layers, n_outputs)

        )


    def forward(self, x):
        # Forward pass 
        return self.model(x)


## number of data points
n_data = 1000

## Number of inputs
n_inputs = 1000
n_hidden_layer_nodes = 100
n_outputs = 10


## Training parameters
n_epochs = 100

## Create input and output tensors
x = torch.randn(n_data, n_inputs)
y = torch.randn(n_data, n_outputs)


## Construct model
model = MLP(n_inputs, n_hidden_layer_nodes, n_outputs)

## Define loss function
loss_function = nn.MSELoss(reduction='sum')  ## Reduction with sum means dividing total loss by n


## Define Optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-4)


for i in range(n_epochs):

    ## Forward pass
    y_pred = model(x)

    ## Compute and print loss
    loss = loss_function(y_pred, y)
    print(i, loss.item())

    ## Zero gradient, perform backward pass, update the weights

    optimizer.zero_grad()

    ## Calculate gradients using backward pass
    loss.backward()

    ## update the model parameters (i.e. weights)
    optimizer.step()






