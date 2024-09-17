# This module handles the loading and processing of image data from a CSV file.
# It includes functionalities for:
# - Initializing the dataset from a CSV file.
# - Processing and normalizing the data, splitting it into training and validation sets.
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

    def process_data(self):
        self.data = np.array(self.data)
        m, n = self.data.shape
        np.random.shuffle(self.data)  # Shuffle all rows

        # Testing data
        data_dev = self.data[:1000].T  # Take the first 1000 rows and transpose
        self.Y_dev = data_dev[0]  # Labels for validation set
        self.X_dev = data_dev[1:]  # Features for validation set
        self.X_dev /= 255.  # Normalize pixel values to [0, 1]

        # Training data
        data_train = self.data[1000:].T  # Take remaining rows
        self.Y_train = data_train[0]  # Labels for training set
        self.X_train = data_train[1:]  # Features for training set
        self.X_train /= 255.  # Normalize pixel values to [0, 1]

    def plot_mnist_image(self, image_array):
        image = image_array.reshape(28, 28)
        plt.imshow(image, cmap='gray')
        plt.axis('off')  # Turn off axis
        plt.show()
