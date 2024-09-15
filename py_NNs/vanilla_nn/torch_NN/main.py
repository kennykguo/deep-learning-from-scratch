# This module trains a neural network model using custom-defined classes and PyTorch. 
# It processes training data, performs forward and backward passes, updates model parameters, and saves the trained model.
# Key functionalities include:
# - Defining a model with Linear and Tanh layers.
# - Training the model using a stochastic gradient descent approach.
# - Processing data from a CSV file and converting labels to one-hot encoding.
# - Saving the trained model parameters to a file.

import torch
import torch.nn.functional as F
from dataset import Dataset
from torch_architecture import Sequential, Linear, Tanh
import os

# Parameters
epochs = 5
batch_size = 64
num_input = 784
num_hidden = 15
num_output = 10
max_iterations = 5000
lr = 0.1

# Define the model using custom classes
Model = Sequential([
    Linear(num_input, num_hidden, bias=True), Tanh(),
    Linear(num_hidden, num_hidden, bias=True), Tanh(),
    Linear(num_hidden, num_output, bias=True),
])

# Enable gradient computation for model parameters
parameters = Model.parameters()
for p in parameters:
    p.requires_grad = True

# Process the data
Data = Dataset('data/train.csv')
Data.process_data()

# Training loop
for epoch in range(max_iterations):
    # Get a random batch of data
    batch = torch.randint(0, Data.X_train.shape[1], (batch_size,))
    Xb, Yb = Data.X_train[batch], Data.Y_train[batch]

    # Convert labels to one-hot encoding
    Yb = Yb.to(torch.int64)
    Yb = F.one_hot(Yb, num_classes=10)
    Yb = Yb.to(torch.float32)

    # Forward pass
    logits = Model(Xb)

    # Calculate loss
    loss = F.cross_entropy(logits, Yb)

    # Backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # Update parameters
    for p in parameters:
        p.data += -lr * p.grad

    # Print the loss every 1000 epochs
    if epoch % 1000 == 0:
        print(f'{epoch:7d}/{max_iterations:7d}: {loss.item():.4f}')

# Save the trained model parameters
directory = os.getcwd() + "/model"
torch.save(Model.parameters(), directory)
