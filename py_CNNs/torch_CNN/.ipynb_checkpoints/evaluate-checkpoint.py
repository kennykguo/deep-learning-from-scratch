import torch
import torch.nn.functional as F
from dataset import Dataset
from torch_architecture import *
import os

# Parameters
batch_size = 32
num_output = 10
input_size = 28
input_depth = 1
output_size = 24
filter_size = 3
kernel_size = 5
max_iterations = 1
hidden_size = int(filter_size * output_size * output_size / 4)
lr = 0.1

Model = Sequential([
    Conv2d(input_size, input_depth, filter_size, kernel_size, batch_size, bias = True), 
    Tanh(), 
    Max_Pooling2d(),
    FlattenConsecutive2d(batch_size), 
    Linear(hidden_size, num_output, bias = True)
    ])

loaded_parameters = torch.load(os.getcwd() + '/model/model.pth')
for param, loaded_param in zip(Model.parameters(), loaded_parameters):
    param.data.copy_(loaded_param.data)


# Process our data
Data = Dataset('data/train.csv')

Data.process_data()

# Put all layers into eval mode
for layer in Model.layers:
    layer.training = False

@torch.no_grad() # this decorator disables gradient tracking
def split_loss(split):
    # Get the dataset
    x,y = {
    'train': (Data.X_train, Data.Y_train),
    'val': (Data.X_dev, Data.Y_dev),
    }[split]

    #Forward pass
    x = x.view(input_size,input_size, 1, -1)
    logits = Model(x)

    # Convert to integers for one hot, then back to float for loss calculations
    y = y.to(torch.int64)
    y = F.one_hot(y, num_classes=10)
    y = y.to(torch.float32)

    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

# split_loss('train')
# split_loss('val')

# Calculate accuracy over 1000 iterations
iterations = 100
counter = 0
for i in range(iterations):
    test = Data.X_dev[:,:,i]
    test = test.view(input_size,input_size, 1, -1)
    label = Data.Y_dev[i]
    logits = Model(test)
    prediction = torch.argmax(logits)
    if (prediction == label):
        counter += 1

# Print the accuracy
print("Accuracy: ")

import torch
import torch.nn.functional as F
from dataset import Dataset
from torch_architecture import *
import os

# Parameters
batch_size = 32
num_output = 10
input_size = 28
input_depth = 1
output_size = 24
filter_size = 3
kernel_size = 5
max_iterations = 1
hidden_size = int(filter_size * output_size * output_size / 4)
lr = 0.1

Model = Sequential([
    Conv2d(input_size, input_depth, filter_size, kernel_size, batch_size, bias = True), 
    Tanh(), 
    Max_Pooling2d(),
    FlattenConsecutive2d(batch_size), 
    Linear(hidden_size, num_output, bias = True)
    ])

loaded_parameters = torch.load(os.getcwd() + '/model/model.pth')
for param, loaded_param in zip(Model.parameters(), loaded_parameters):
    param.data.copy_(loaded_param.data)


# Process our data
Data = Dataset('data/train.csv')

Data.process_data()

# Put all layers into eval mode
for layer in Model.layers:
    layer.training = False

@torch.no_grad() # this decorator disables gradient tracking
def split_loss(split):
    # Get the dataset
    x,y = {
    'train': (Data.X_train, Data.Y_train),
    'val': (Data.X_dev, Data.Y_dev),
    }[split]

    #Forward pass
    x = x.view(input_size,input_size, 1, -1)
    logits = Model(x)

    # Convert to integers for one hot, then back to float for loss calculations
    y = y.to(torch.int64)
    y = F.one_hot(y, num_classes=10)
    y = y.to(torch.float32)

    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

# split_loss('train')
# split_loss('val')

# Calculate accuracy over 1000 iterations
iterations = 100
counter = 0
for i in range(iterations):
    test = Data.X_dev[:,:,i]
    test = test.view(input_size,input_size, 1, -1)
    label = Data.Y_dev[i]
    logits = Model(test)
    prediction = torch.argmax(logits)
    if (prediction == label):
        counter += 1

# Print the accuracy
print("Accuracy: ")
print((counter / iterations) * 100 , "%")