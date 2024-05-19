import Dataset
import Network
import numpy as np
import pandas as pd
from Dataset import Dataset
import os


Data = Dataset('train.csv')
Data.process_data()
print(Data.X_train.shape)