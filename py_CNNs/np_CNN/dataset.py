# This module handles loading and processing of MNIST-like image data from a CSV file.
# It includes functionalities for:
# - Initializing the dataset from a CSV file.
# - Processing and splitting the data into training, validation, and test sets.
# - Reshaping data into image format.
# - Plotting individual MNIST images.

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class Dataset:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.X_dev = None
        self.Y_dev = None
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None

    def process_data(self):
        self.data = np.array(self.data)
        m, n = self.data.shape

        np.random.shuffle(self.data)  # Shuffle all rows

        data_dev = self.data[0:1000].T  # Take the first 1000 rows and transpose
        self.Y_dev = data_dev[0]  # Labels for validation set
        self.X_dev = data_dev[1:n]  # Features for validation set
        self.X_dev = self.X_dev.reshape(28, 28, 1000)  # Reshape to image format

        data_train = self.data[2000:m].T  # Take rows from 2000 to end
        self.Y_train = data_train[0]  # Labels for training set
        self.X_train = data_train[1:n]  # Features for training set
        self.X_train = self.X_train.reshape(28, 28, -1)  # Reshape to image format

        data_test = self.data[1000:2000].T  # Take rows from 1000 to 2000
        self.Y_test = data_test[0]  # Labels for test set
        self.X_test = data_test[1:n]  # Features for test set
        self.X_test = self.X_test.reshape(28, 28, -1)  # Reshape to image format

    def plot_mnist_image(self, image_array):
        image = image_array.reshape(28, 28)
        plt.imshow(image, cmap='gray')
        plt.axis('off')  # Turn off axis
        plt.show()
