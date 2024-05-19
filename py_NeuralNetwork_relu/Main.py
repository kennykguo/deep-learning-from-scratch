from Network import Network
from Dataset import Dataset
import numpy as np
import pandas as pd
import os

# Interface
Data = Dataset('train.csv')
Data.process_data()
Model = Network()
Model.stochastic_gradient_descent(Data.X_train, Data.Y_train, Data.X_dev, Data.Y_dev, 30, 50, 0.01)


np.savez("model_parameters.npz", Model.dW1, Model.b1, dW2=Model.W2, db2=Model.db2)