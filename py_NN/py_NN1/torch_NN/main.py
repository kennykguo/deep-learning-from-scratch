import torch
import torch.nn.functional as F
from dataset import Dataset
from torch_NN import Sequential, Linear, Tanh


# Parameter dimensions
epochs = 1
batch_size = 32
num_input = 784
num_hidden = 15
num_output = 10
max_iterations = 20000
lr = 0.1

# Define the model using our classes
Model = Sequential([
    Linear(num_input, num_hidden, bias=True), Tanh(),
    Linear(num_hidden, num_hidden, bias=True), Tanh(),
    Linear(num_hidden, num_output, bias=True),
])

# Turn on requires_grad to use loss.backward()
parameters = Model.parameters()
for p in parameters:
  p.requires_grad = True

Data = Dataset('data/train.csv')
Data.process_data()

batch = torch.randint(0, Data.X_train.shape[1], (batch_size,)) #(784, 40000)

for epoch in range(max_iterations):
    # Get a random batch
    batch = torch.randint(0, Data.X_train.shape[1], (batch_size,)) #(784, 40000)
    Xb, Yb = Data.X_train[batch], Data.Y_train[batch]

    # Need to do this because torch only accepts integer one hots
    Yb = Yb.to(torch.int64)
    Yb = F.one_hot(Yb, num_classes=10)
    Yb = Yb.to(torch.float32)

    # Forward pass
    logits = Model(Xb)

    # Calculate the loss
    loss = F.cross_entropy(logits, Yb)

    # Backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # Update parameters
    for p in parameters:
        p.data += -lr * p.grad

    # Print the loss
    if epoch % 500 == 0: # print every once in a while
        print(f'{epoch:7d}/{max_iterations:7d}: {loss.item():.4f}')