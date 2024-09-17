"""
This script performs evaluation on a trained convolutional neural network (CNN) model using PyTorch. It includes functionality for loading the model parameters, processing data, and computing loss and accuracy metrics.

Functionality:
- Load the model architecture and trained parameters.
- Process the dataset for training and validation.
- Evaluate the model using cross-entropy loss on both training and validation datasets.
- Calculate the accuracy of the model on the validation dataset over a specified number of iterations.
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

# Define the model architecture
Model = Sequential([
    Conv2d(input_size, input_depth, filter_size, kernel_size, batch_size, bias=True),
    Tanh(),
    Max_Pooling2d(),
    FlattenConsecutive2d(batch_size),
    Linear(hidden_size, num_output, bias=True)
])

# Load trained parameters into the model
loaded_parameters = torch.load(os.getcwd() + '/model/model.pth')
for param, loaded_param in zip(Model.parameters(), loaded_parameters):
    param.data.copy_(loaded_param.data)

# Process the data
Data = Dataset('data/train.csv')
Data.process_data()

# Set all layers to evaluation mode
for layer in Model.layers:
    layer.training = False

@torch.no_grad()  # This decorator disables gradient tracking
def split_loss(split):
    # Get the dataset for the specified split
    x, y = {
        'train': (Data.X_train, Data.Y_train),
        'val': (Data.X_dev, Data.Y_dev),
    }[split]

    # Forward pass
    x = x.view(input_size, input_size, 1, -1)
    logits = Model(x)

    # Convert labels to one-hot encoding for loss calculation
    y = y.to(torch.int64)
    y = F.one_hot(y, num_classes=10)
    y = y.to(torch.float32)

    # Compute and print loss
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

# Uncomment to print loss for training and validation sets
# split_loss('train')
# split_loss('val')

# Calculate accuracy over a number of iterations
iterations = 100
counter = 0
for i in range(iterations):
    test = Data.X_dev[:, :, i]
    test = test.view(input_size, input_size, 1, -1)
    label = Data.Y_dev[i]
    logits = Model(test)
    prediction = torch.argmax(logits)
    if prediction == label:
        counter += 1

# Print the accuracy
print("Accuracy: ")
print((counter / iterations) * 100, "%")
