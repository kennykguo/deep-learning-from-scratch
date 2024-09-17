"""
This module defines a `Dataset` class for managing and processing data from a CSV file for machine learning tasks. It provides methods to load, preprocess, and convert data into PyTorch tensors, as well as visualize MNIST images.

Functionality:
- `__init__(file_path)`: Initializes the dataset from a CSV file.
- `process_data()`: Processes the data by shuffling, splitting into training, validation, and test sets, reshaping, and converting it to PyTorch tensors.
- `plot_mnist_image(image_array)`: Visualizes a 28x28 image from a given array using matplotlib.

Usage:
1. Create an instance of the `Dataset` class with the path to the CSV file.
2. Call `process_data()` to prepare the dataset.
3. Use `plot_mnist_image()` to display MNIST images.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch

class Dataset:
    def __init__(self, file_path):
        # Load the data from a CSV file
        self.data = pd.read_csv(file_path)
        self.X_dev = None
        self.Y_dev = None
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None

    def process_data(self):
        # Convert the data to a NumPy array
        self.data = np.array(self.data)
        m, n = self.data.shape

        # Shuffle the data
        np.random.shuffle(self.data)

        # Split and reshape the data for development (validation) set
        data_dev = self.data[0:1000].T  # Take the first 1000 rows, transpose to get column vectors
        self.Y_dev = data_dev[0]  # First row contains labels (Y)
        self.X_dev = data_dev[1:n]
        self.X_dev = self.X_dev.reshape(28, 28, 1000)

        # Split and reshape the data for training set
        data_train = self.data[2000:m].T
        self.Y_train = data_train[0]
        self.X_train = data_train[1:n]
        self.X_train = self.X_train.reshape(28, 28, -1)

        # Split and reshape the data for test set
        data_test = self.data[1000:2000].T
        self.Y_test = data_test[0]
        self.X_test = data_test[1:n]
        self.X_test = self.X_test.reshape(28, 28, -1)

        # Convert the data to PyTorch tensors
        self.X_dev = torch.from_numpy(self.X_dev).to(torch.float32)
        self.Y_dev = torch.from_numpy(self.Y_dev).to(torch.float32)
        self.X_train = torch.from_numpy(self.X_train).to(torch.float32)
        self.Y_train = torch.from_numpy(self.Y_train).to(torch.float32)
        self.X_test = torch.from_numpy(self.X_test).to(torch.float32)
        self.Y_test = torch.from_numpy(self.Y_test).to(torch.float32)

    def plot_mnist_image(self, image_array):
        # Reshape and display a 28x28 image
        image = image_array.reshape(28, 28)
        plt.imshow(image, cmap='gray')
        plt.axis('off')  # Turn off axis
        plt.show()
