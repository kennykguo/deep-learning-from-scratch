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

# (28, 28, 1, 32) -> (24, 24, 12, 32) -> (12, 12 12, 32) -> (32, 1728) -> (32, 10)

# Define the model using our classes
Model = Sequential([
    Conv2d(input_size, input_depth, filter_size, kernel_size, batch_size, bias = True), 
    Tanh(), 
    Max_Pooling2d(),
    FlattenConsecutive2d(batch_size), 
    Linear(hidden_size, num_output, bias = True)
    ])

# Turn on requires_grad to use loss.backward()
parameters = Model.parameters()
for p in parameters:
  p.requires_grad = True

# Process our data
Data = Dataset('data/train.csv')

Data.process_data()

# Training loop
for epoch in range(max_iterations):
    # Get a random batch
    batch = torch.randint(0, Data.X_train.shape[2], (batch_size,))
    Xb, Yb = Data.X_train[:,:, batch], Data.Y_train[batch]
    Xb = Xb.view(input_size,input_size, 1, batch_size)

    # Need to do this because torch only accepts integer one hots
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


directory = os.getcwd() + "model"

torch.save(Model.parameters(), directory)