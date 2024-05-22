import numpy as np
import scipy as signal
import math
import torch
import numpy as np
# import pandas as pd
import os

from CNNNetwork import ConvolutionalNN
from Dataset import Dataset

# np.random.seed(seed=42)

# Interface
Data = Dataset('data/train.csv')

Data.process_data()

Model = ConvolutionalNN()

# Epochs, lr, dropout, batch
Model.stochastic_gradient_descent(Data.X_train, Data.X_dev, Data.Y_train, Data.Y_dev, 10, 0.2, 0.5, 32)

# np.savez("model_parameters.npz", Model.dW1, Model.b1, dW2=Model.W2, db2=Model.db2)

Model.plot_accuracies()

