import torch
import torch.nn as nn

# Module: SimpleLSTM
# This module defines a simple LSTM-based model for sequence prediction.
# The model consists of an LSTM layer followed by a fully connected layer.
# It includes functions for training the model, evaluating on a development set,
# and converting sequences to one-hot vectors.
# Usage:
# 1. Define the model parameters and instantiate the SimpleLSTM class.
# 2. Prepare and preprocess data using helper functions.
# 3. Train the model using the training loop.
# 4. Evaluate the model on the development set periodically.

batch_size = 32
hidden_size = 30
lr = 0.01
time_steps = 8
input_size = 27
output_size = 27

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the SimpleLSTM model.
        
        Args:
        input_size (int): Size of the input feature vector.
        hidden_size (int): Number of hidden units in the LSTM layer.
        output_size (int): Size of the output vector.
        """
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0, c0):
        """
        Forward pass through the LSTM and fully connected layer.
        
        Args:
        x (Tensor): Input tensor of shape (batch_size, time_steps, input_size).
        h0 (Tensor): Initial hidden state.
        c0 (Tensor): Initial cell state.
        
        Returns:
        Tuple: Output tensor and the final hidden and cell states.
        """
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out, (hn, cn)

# Instantiate the model, loss function, and optimizer
model = SimpleLSTM(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

iterations = 0
num_epochs = 2000

# Initialize hidden and cell states
h0 = torch.zeros(1, batch_size, hidden_size)
c0 = torch.zeros(1, batch_size, hidden_size)

def to_one_hot(indices, num_classes):
    """
    Converts indices to one-hot vectors.
    
    Args:
    indices (Tensor): Tensor of shape (batch_size, time_steps) with class indices.
    num_classes (int): Number of classes.
    
    Returns:
    Tensor: One-hot encoded tensor of shape (batch_size, time_steps, num_classes).
    """
    one_hot = torch.zeros(indices.size(0), indices.size(1), num_classes, device=indices.device)
    one_hot.scatter_(2, indices.unsqueeze(2), 1.0)
    return one_hot

# Convert training and development data to one-hot vectors
one_hot_Xtr = to_one_hot(Xtr_batched, input_size)
one_hot_Xdev = to_one_hot(Xdev_batched, input_size)

# Training loop
for epoch in range(num_epochs):
    loss = 0
    for i in range(0, one_hot_Xtr.size(1) - time_steps, time_steps):
        Xb = one_hot_Xtr[:, i:i + time_steps, :]
        Yb = Ytr_batched[:, i:i + time_steps]

        optimizer.zero_grad()
        
        output, (h0, c0) = model(Xb, h0.detach(), c0.detach())
        
        loss = criterion(output.view(-1, output_size), Yb.view(-1))
        loss.backward()
        
        optimizer.step()
        
        if (i // time_steps) % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Step [{i // time_steps}], Loss: {loss.item():.4f}')

    # Evaluate on development set
    if (epoch + 1) % 100 == 0:
        with torch.no_grad():
            dev_loss = 0
            for j in range(0, one_hot_Xdev.size(1) - time_steps, time_steps):
                Xb_dev = one_hot_Xdev[:, j:j + time_steps, :]
                Yb_dev = Ydev_batched[:, j:j + time_steps]
                
                output_dev, _ = model(Xb_dev, h0, c0)
                dev_loss += criterion(output_dev.view(-1, output_size), Yb_dev.view(-1)).item()
            print(f'Epoch [{epoch + 1}/{num_epochs}], Dev Loss: {dev_loss / (one_hot_Xdev.size(1) // time_steps):.4f}')

    iterations += 1
