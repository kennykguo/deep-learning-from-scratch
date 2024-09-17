"""
This module trains a neural network model using a custom architecture defined in `torch_architecture`. It handles data preprocessing, model training, and parameter saving. 

Functionality:
- Defines and initializes the model using custom classes.
- Loads and processes training data from a CSV file.
- Performs training over a specified number of iterations, including forward and backward passes.
- Updates model parameters and saves the trained model.

Usage:
1. Define and initialize your model.
2. Load and preprocess the data.
3. Run the training loop for a specified number of epochs.
4. Save the trained model parameters.
"""

import torch
import torch.nn.functional as F
from dataset import Dataset
from torch_architecture import *
import os

# Parameters
batch_size = 32
num_output = 10
input_size = 28
input_depth = 1
output_size = 24
filter_size = 3
kernel_size = 5
max_iterations = 1
hidden_size = int(filter_size * output_size * output_size / 4)
lr = 0.1

# Define the model using custom classes
Model = Sequential([
    Conv2d(input_size, input_depth, filter_size, kernel_size, batch_size, bias=True),
    Tanh(),
    Max_Pooling2d(),
    FlattenConsecutive2d(batch_size),
    Linear(hidden_size, num_output, bias=True)
])

# Enable gradient computation for all parameters
parameters = Model.parameters()
for p in parameters:
    p.requires_grad = True

# Process the training data
Data = Dataset('data/train.csv')
Data.process_data()

# Training loop
for epoch in range(max_iterations):
    # Get a random batch
    batch = torch.randint(0, Data.X_train.shape[2], (batch_size,))
    Xb, Yb = Data.X_train[:, :, batch], Data.Y_train[batch]
    Xb = Xb.view(input_size, input_size, 1, batch_size)

    # Convert labels to one-hot encoding
    Yb = Yb.to(torch.int64)
    Yb = F.one_hot(Yb, num_classes=10)
    Yb = Yb.to(torch.float32)

    # Forward pass
    logits = Model(Xb)

    # Calculate the loss
    loss = F.cross_entropy(logits, Yb)

    # Backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # Update parameters
    for p in parameters:
        p.data += -lr * p.grad

    # Print the loss
    print(f'{epoch:7d}/{max_iterations:7d}: {loss.item():.4f}')

# Save the trained model parameters
directory = os.getcwd() + "/model"
torch.save(Model.parameters(), 'model.pth')
