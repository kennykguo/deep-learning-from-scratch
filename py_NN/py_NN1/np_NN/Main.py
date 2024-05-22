from np_NN import Network
from dataset import Dataset
import torch
import numpy as np
import pandas as pd
import os

# Interface
Data = Dataset('data/train.csv')

Data.process_data()

Model = Network()

Model.stochastic_gradient_descent(Data.X_train, Data.Y_train, Data.X_dev, Data.Y_dev, 5, 0.2, 0.5, 32)

Model.plot_accuracies()
