"""
This module implements a Simple Recurrent Neural Network (RNN) for character-level language modeling.

Functions:
1. `random_init(num_rows, num_cols)`: Initializes a matrix with random values scaled by 0.01.
2. `zero_init(num_rows, num_cols)`: Initializes a matrix with zeros.
3. `DataReader`: A class for reading and processing text data. It handles:
   - Initialization with the text file path and sequence length.
   - Generating the next batch of inputs and targets.
   - Checking if the reading has just started.
   - Closing the file.
4. `SimpleRNN`: A class for the Simple RNN model. It includes:
   - Initialization with hidden size, vocab size, sequence length, and learning rate.
   - Forward pass through the network.
   - Backward pass to compute gradients.
   - Loss calculation.
   - Model parameter updates using AdaGrad.
   - Sequence sampling from the model.
   - Training loop for model training and sampling.

Usage:
- Create an instance of `DataReader` with the path to your text file and the desired sequence length.
- Create an instance of `SimpleRNN` with appropriate hyperparameters.
- Call the `train` method of `SimpleRNN` with the `DataReader` instance to start training.
"""

import numpy as np

def random_init(num_rows, num_cols):
    """
    Initializes a matrix with random values scaled by 0.01.
    
    Args:
    num_rows (int): Number of rows in the matrix.
    num_cols (int): Number of columns in the matrix.
    
    Returns:
    np.ndarray: Initialized matrix.
    """
    return np.random.rand(num_rows, num_cols) * 0.01

def zero_init(num_rows, num_cols):
    """
    Initializes a matrix with zeros.
    
    Args:
    num_rows (int): Number of rows in the matrix.
    num_cols (int): Number of columns in the matrix.
    
    Returns:
    np.ndarray: Initialized matrix.
    """
    return np.zeros((num_rows, num_cols))

class DataReader:
    """
    A class for reading and processing text data. Handles text encoding and data batching.
    """
    def __init__(self, path, seq_length):
        """
        Initializes the DataReader with a file path and sequence length.
        
        Args:
        path (str): Path to the text file.
        seq_length (int): Length of the sequences for training.
        """
        self.fp = open(path, "r")
        self.data = self.fp.read()
        chars = list(set(self.data))
        self.char_to_ix = {ch: i for (i, ch) in enumerate(chars)}
        self.ix_to_char = {i: ch for (i, ch) in enumerate(chars)}
        self.data_size = len(self.data)
        self.vocab_size = len(chars)
        self.pointer = 0
        self.seq_length = seq_length

    def next_batch(self):
        """
        Retrieves the next batch of input and target sequences.
        
        Returns:
        tuple: A tuple containing the input and target sequences as lists of indices.
        """
        input_start = self.pointer
        input_end = self.pointer + self.seq_length
        inputs = [self.char_to_ix[ch] for ch in self.data[input_start:input_end]]
        targets = [self.char_to_ix[ch] for ch in self.data[input_start+1:input_end+1]]
        self.pointer += self.seq_length
        if self.pointer + self.seq_length + 1 >= self.data_size:
            # Reset pointer if end of data is reached
            self.pointer = 0
        return inputs, targets

    def just_started(self):
        """
        Checks if the data reading has just started.
        
        Returns:
        bool: True if the pointer is at the start, otherwise False.
        """
        return self.pointer == 0

    def close(self):
        """
        Closes the data file.
        """
        self.fp.close()

class SimpleRNN:
    """
    A class for the Simple Recurrent Neural Network model.
    """
    def __init__(self, hidden_size, vocab_size, seq_length, learning_rate):
        """
        Initializes the SimpleRNN model with hyperparameters and model parameters.
        
        Args:
        hidden_size (int): Number of hidden units.
        vocab_size (int): Size of the vocabulary.
        seq_length (int): Length of the input sequences.
        learning_rate (float): Learning rate for training.
        """
        # Hyperparameters
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate

        # Model parameters
        self.Wxh = random_init(hidden_size, vocab_size)  # Input to hidden weights
        self.Whh = random_init(hidden_size, hidden_size)  # Hidden to hidden weights
        self.Why = random_init(vocab_size, hidden_size)  # Hidden to output weights
        self.bh = zero_init(hidden_size, 1)  # Bias for hidden layer
        self.by = zero_init(vocab_size, 1)  # Bias for output layer

        # Memory variables for AdaGrad
        self.mWxh = np.zeros_like(self.Wxh)
        self.mWhh = np.zeros_like(self.Whh)
        self.mWhy = np.zeros_like(self.Why)
        self.mbh = np.zeros_like(self.bh)
        self.mby = np.zeros_like(self.by)

    def forward(self, inputs, hprev):
        """
        Performs the forward pass of the RNN.
        
        Args:
        inputs (list): List of input indices.
        hprev (np.ndarray): Previous hidden state.
        
        Returns:
        tuple: A tuple containing input activations, hidden states, and probabilities for each timestep.
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        for t in range(len(inputs)):
            xs[t] = zero_init(self.vocab_size, 1)
            xs[t][inputs[t]] = 1  # One-hot encoding
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh)  # Hidden state
            ys[t] = np.dot(self.Why, hs[t]) + self.by  # Unnormalized log probabilities for next char
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # Probabilities for next char
        return xs, hs, ps

    def backward(self, xs, hs, ps, targets):
        """
        Computes gradients for the backward pass.
        
        Args:
        xs (dict): Input activations.
        hs (dict): Hidden states.
        ps (dict): Probabilities for each timestep.
        targets (list): Target indices.
        
        Returns:
        tuple: Gradients for input-to-hidden, hidden-to-hidden, hidden-to-output weights, and biases.
        """
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])
        for t in reversed(range(self.seq_length)):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1  # Backprop into y
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dhnext  # Backprop into h
            dhraw = (1 - hs[t] * hs[t]) * dh  # Backprop through tanh non-linearity
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            dhnext = np.dot(self.Whh.T, dhraw)
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)  # Clip to mitigate exploding gradients
        return dWxh, dWhh, dWhy, dbh, dby

    def loss(self, ps, targets):
        """
        Computes the loss for a sequence.
        
        Args:
        ps (dict): Probabilities for each timestep.
        targets (list): Target indices.
        
        Returns:
        float: Loss value.
        """
        return sum(-np.log(ps[t][targets[t], 0]) for t in range(self.seq_length))

    def update_model(self, dWxh, dWhh, dWhy, dbh, dby):
        """
        Updates model parameters using AdaGrad.
        
        Args:
        dWxh (np.ndarray): Gradient for input-to-hidden weights.
        dWhh (np.ndarray): Gradient for hidden-to-hidden weights.
        dWhy (np.ndarray): Gradient for hidden-to-output weights.
        dbh (np.ndarray): Gradient for hidden layer bias.
        dby (np.ndarray): Gradient for output bias.
        """
        for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
                                      [dWxh, dWhh, dWhy, dbh, dby],
                                      [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]):
            mem += dparam * dparam
            param -= self.learning_rate * dparam / (np.sqrt(mem) + 1e-8)

    def sample(self, hprev, seed_ix, n):
        """
        Samples a sequence of characters from the model.
        
        Args:
        hprev (np.ndarray): Initial hidden state.
        seed_ix (int): Initial character index.
        n (int): Length of the sequence to sample.
        
        Returns:
        list: Sampled sequence of characters.
        """
        x = zero_init(self.vocab_size, 1)
        x[seed_ix] = 1
        idxes = []
        for t in range(n):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, hprev) + self.bh)
            y = np.dot(self.Why, h) + self.by
            p = np.exp(y) / np.sum(np.exp(y))
            idx = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = zero_init(self.vocab_size, 1)
            x[idx] = 1
            idxes.append(idx)
        return idxes

    def train(self, data_reader, num_epochs):
        """
        Trains the RNN model.
        
        Args:
        data_reader (DataReader): An instance of the DataReader class.
        num_epochs (int): Number of training epochs.
        """
        for epoch in range(num_epochs):
            if data_reader.just_started():
                hprev = zero_init(self.hidden_size, 1)
            inputs, targets = data_reader.next_batch()
            xs, hs, ps = self.forward(inputs, hprev)
            dWxh, dWhh, dWhy, dbh, dby = self.backward(xs, hs, ps, targets)
            self.update_model(dWxh, dWhh, dWhy, dbh, dby)
            loss = self.loss(ps, targets)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')
            if epoch % 1000 == 0:
                sample_ix = self.sample(hprev, inputs[0], 200)
                sample_text = ''.join(data_reader.ix_to_char[ix] for ix in sample_ix)
                print(f'Sample: {sample_text}')

# Example usage:
# data_reader = DataReader('path/to/text.txt', seq_length=25)
# model = SimpleRNN(hidden_size=100, vocab_size=len(data_reader.char_to_ix), seq_length=25, learning_rate=1e-1)
# model.train(data_reader, num_epochs=10000)
