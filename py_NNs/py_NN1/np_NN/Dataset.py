import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class Dataset:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.X_dev = None;
        self.Y_dev = None;
        self.X_train = None;
        self.Y_train = None;

    def process_data(self):
        self.data = np.array(self.data)
        m, n = self.data.shape
        np.random.shuffle(self.data)
        # Testing data
        data_dev = self.data[0:1000].T #Take the first 1000 rows, and transpose the matrix to get 1000 examples as column vectors
        self.Y_dev = data_dev[0]
        self.X_dev = data_dev[1:] #Takes all of the data corresponding to all of the entries (the X values)
        self.X_dev = self.X_dev / 255.
        # Training data
        data_train = self.data[1000:m].T
        self.Y_train = data_train[0]
        self.X_train= data_train[1:n] #Takes all of the data corresponding to all of the entries
        self.X_train = self.X_train / 255.

    def plot_mnist_image(self, image_array):
        image = image_array.reshape(28, 28)
        plt.imshow(image, cmap='gray')
        plt.axis('off')  # Turn off axis
=======
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class Dataset:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.X_dev = None;
        self.Y_dev = None;
        self.X_train = None;
        self.Y_train = None;

    def process_data(self):
        self.data = np.array(self.data)
        m, n = self.data.shape
        np.random.shuffle(self.data)
        # Testing data
        data_dev = self.data[0:1000].T #Take the first 1000 rows, and transpose the matrix to get 1000 examples as column vectors
        self.Y_dev = data_dev[0]
        self.X_dev = data_dev[1:] #Takes all of the data corresponding to all of the entries (the X values)
        self.X_dev = self.X_dev / 255.
        # Training data
        data_train = self.data[1000:m].T
        self.Y_train = data_train[0]
        self.X_train= data_train[1:n] #Takes all of the data corresponding to all of the entries
        self.X_train = self.X_train / 255.

    def plot_mnist_image(self, image_array):
        image = image_array.reshape(28, 28)
        plt.imshow(image, cmap='gray')
        plt.axis('off')  # Turn off axis
        plt.show()