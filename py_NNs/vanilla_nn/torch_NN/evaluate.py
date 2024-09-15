# This module defines a neural network model using PyTorch. It loads a pretrained model, processes data from a CSV file, and calculates both loss and accuracy for the training and validation datasets.
# Key functionalities include:
# - Defining a sequential neural network model with Linear and Tanh layers.
# - Loading pretrained model parameters from a .pth file.
# - Processing training data using a custom Dataset class.
# - Disabling gradient tracking for evaluation using torch.no_grad().
# - Computing cross-entropy loss for training and validation datasets.
# - Calculating model accuracy over a set number of iterations.

import torch
import torch.nn.functional as F
from dataset import Dataset
from py_NN.py_NN1.torch_NN.torch_architecture import Sequential, Linear, Tanh
import os

# Parameter dimensions
epochs = 1
batch_size = 32
num_input = 784
num_hidden = 15
num_output = 10
max_iterations = 10000
lr = 0.1

# Define the Sequential model using Linear layers and Tanh activations
Model = Sequential([
    Linear(num_input, num_hidden, bias=True), Tanh(),
    Linear(num_hidden, num_hidden, bias=True), Tanh(),
    Linear(num_hidden, num_output, bias=True),
])

# Load pretrained model parameters
loaded_parameters = torch.load(os.getcwd() + '/model/model.pth')
for param, loaded_param in zip(Model.parameters(), loaded_parameters):
    param.data.copy_(loaded_param.data)

# Process the training data using a custom Dataset class
Data = Dataset('data/train.csv')
Data.process_data()

# Set the model to evaluation mode by disabling training mode for each layer
for layer in Model.layers:
    layer.training = False

@torch.no_grad()  # This decorator disables gradient tracking during evaluation
def split_loss(split):
    # Get the dataset (train or validation)
    x, y = {
        'train': (Data.X_train, Data.Y_train),
        'val': (Data.X_dev, Data.Y_dev),
    }[split]

    # Perform the forward pass
    logits = Model(x)

    # Convert labels to one-hot encoding for loss computation
    y = y.to(torch.int64)
    y = F.one_hot(y, num_classes=10)
    y = y.to(torch.float32)

    # Calculate cross-entropy loss
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

# Compute loss for both training and validation sets
split_loss('train')
split_loss('val')

# Calculate accuracy over 1000 iterations on the validation set
iterations = 1000
counter = 0
for i in range(iterations):
    test = Data.X_dev[i]
    label = Data.Y_dev[i]
    logits = Model(test)
    prediction = torch.argmax(logits)
    if prediction == label:
        counter += 1

# Print the accuracy percentage
print("Accuracy: ")
print((counter / iterations) * 100, "%")
