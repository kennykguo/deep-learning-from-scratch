# This module handles the training of a Convolutional Neural Network (CNN) on image data.
# It includes functionalities for:
# - Loading and processing data from a CSV file using the Dataset class.
# - Training a CNN model using stochastic gradient descent.
# - (Commented out) Saving model parameters to a file.
# - (Commented out) Plotting model accuracies.

import numpy as np
import scipy as signal
import math
import torch
import os
import matplotlib.pyplot as plt

from CNNNetwork import ConvolutionalNN
from py_CNNs.np_CNN.dataset import Dataset

np.random.seed(42)

# Load and process data
Data = Dataset('data/train.csv')
Data.process_data()

# Initialize the model
Model = ConvolutionalNN()

# Train the model using stochastic gradient descent
Model.stochastic_gradient_descent(Data.X_train, Data.X_dev, Data.Y_train, Data.Y_dev, 1, 0.2, 0.5, 32)

# Save model parameters (commented out)
# np.savez("model_parameters.npz", Model.dW1, Model.b1, dW2=Model.W2, db2=Model.db2)

# Plot model accuracies (commented out)
# Model.plot_accuracies()
