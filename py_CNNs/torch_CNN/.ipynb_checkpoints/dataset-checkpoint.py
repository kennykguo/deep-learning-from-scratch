import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch

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

        np.random.shuffle(self.data) # Shuffles all the individual rows

        data_dev = self.data[0:1000].T #Take the first 1000 rows, and transpose the matrix to get 1000 examples as column vectors
        self.Y_dev = data_dev[0] #Takes the first row, which contains all of the answers to the numbers (the Y is what we want)
        self.X_dev = data_dev[1:n]
        self.X_dev = self.X_dev.reshape(28,28,1000)

        data_train = self.data[2000:m].T
        self.Y_train = data_train[0]
        self.X_train= data_train[1:n] #Takes all of the data corresponding to all of the entries
        self.X_train = self.X_train.reshape(28,28,-1)

        data_test = self.data[1000:2000].T
        self.Y_test = data_test[0]
        self.X_test= data_test[1:n] #Takes all of the data corresponding to all of the entries
        self.X_test = self.X_test.reshape(28,28,-1)

        self.X_dev = torch.from_numpy(self.X_dev).to(torch.float32)
        self.Y_dev = torch.from_numpy(self.Y_dev).to(torch.float32)
        self.X_train = torch.from_numpy(self.X_train).to(torch.float32)
        self.Y_train = torch.from_numpy(self.Y_train).to(torch.float32)
        self.X_test = torch.from_numpy(self.X_test).to(torch.float32)
        self.Y_test = torch.from_numpy(self.Y_test).to(torch.float32)


    def plot_mnist_image(self, image_array):
        image = image_array.reshape(28, 28)
        plt.imshow(image, cmap='gray')
        plt.axis('off')  # Turn off axis
        plt.show()
=======
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch

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

        np.random.shuffle(self.data) # Shuffles all the individual rows

        data_dev = self.data[0:1000].T #Take the first 1000 rows, and transpose the matrix to get 1000 examples as column vectors
        self.Y_dev = data_dev[0] #Takes the first row, which contains all of the answers to the numbers (the Y is what we want)
        self.X_dev = data_dev[1:n]
        self.X_dev = self.X_dev.reshape(28,28,1000)

        data_train = self.data[2000:m].T
        self.Y_train = data_train[0]
        self.X_train= data_train[1:n] #Takes all of the data corresponding to all of the entries
        self.X_train = self.X_train.reshape(28,28,-1)

        data_test = self.data[1000:2000].T
        self.Y_test = data_test[0]
        self.X_test= data_test[1:n] #Takes all of the data corresponding to all of the entries
        self.X_test = self.X_test.reshape(28,28,-1)

        self.X_dev = torch.from_numpy(self.X_dev).to(torch.float32)
        self.Y_dev = torch.from_numpy(self.Y_dev).to(torch.float32)
        self.X_train = torch.from_numpy(self.X_train).to(torch.float32)
        self.Y_train = torch.from_numpy(self.Y_train).to(torch.float32)
        self.X_test = torch.from_numpy(self.X_test).to(torch.float32)
        self.Y_test = torch.from_numpy(self.Y_test).to(torch.float32)


    def plot_mnist_image(self, image_array):
        image = image_array.reshape(28, 28)
        plt.imshow(image, cmap='gray')
        plt.axis('off')  # Turn off axis
        plt.show()
        