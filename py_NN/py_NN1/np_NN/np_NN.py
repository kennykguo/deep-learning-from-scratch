import numpy as np
import math
import matplotlib.pyplot as plt

class Network:
    def __init__(self):
        # Parameters
        self.W1 = np.random.rand(15, 784)
        # Initialize weights based on Kaiming init paper
        self.W1 *= (math.sqrt(2)) * math.sqrt(1/784)
        # Initialize biases to zero
        self.b1 = np.random.rand(15, 1)
        self.b1 *= 0;
        self.W2 = np.random.rand(10, 15)
        self.W2 *= (math.sqrt(2)) * math.sqrt(1/15)
        self.b2 = np.random.rand(10, 1)
        self.b2 *= 0
        # Gradients
        self.dW1 = np.random.rand(15, 784)
        self.dW1 *= 0
        self.db1 = np.random.rand(15, 1)
        self.db1 *= 0
        self.dW2 = np.random.rand(10, 15)
        self.dW2 *= 0
        self.db2 = np.random.rand(10, 1)
        self.db2 *= 0
        self.accuracies = []
        np.random.seed(seed=42)
    
    def ReLU(self, Z): # Takes in a scalar, returns a scalar
        return np.maximum(Z, 0)

    def softmax(self, Z):
        # Apply softmax column-wise
        exp_Z = np.exp(Z - np.max(Z, axis=0))  # Subtracting the maximum value in each column to avoid overflow
        return exp_Z / np.sum(exp_Z, axis=0)


    def der_ReLU(self, Z):
        return Z > 0

    def create(self, Y): # Passing in a 1 x 41000 matrix (41000 columns, 1 row)
        matrix_Y = np.zeros((Y.size, 10)) # 41000 x 10
        matrix_Y[np.arange(Y.size), Y] = 1 # array([    0,     1,     2, ..., 40997, 40998, 40999]), then indexing the Y values aswell at each column, changing that value to 1
        matrix_Y = matrix_Y.T # 10 x 41000
        return matrix_Y

    def forward_prop(self, X):
        Z1 = self.W1.dot(X) + self.b1 # (15 x 784) (784 x 32) + (15 x 1) -> (15 x 32)
        A1 = self.ReLU(Z1)
        Z2 = self.W2.dot(A1) + self.b2 # (10 x 15) (15 x 32) + (10 x 1) = (10 x 32)
        A2 = self.softmax(Z2)
        # print("Forward ", self.b2.shape)
        return Z1, A1, Z2, A2

    def back_prop(self, Z1, A1, A2, X, Y, batch_size):
        mat_Y = self.create(Y)
        dZ2 = (1 / batch_size) * (A2 - mat_Y) # (10 x 32) - - - Back propogation eq. #1
        self.dW2 = dZ2.dot(A1.T) # (10 x 32) (32 x 15) -> (10 x 15) - - - Back propogation eq. #4
        self.db2 = np.sum(dZ2, axis = 1) # Back propogation eq. #3 - we are summing over cols in the backprop
        self.db2 = self.db2.reshape(10, 1)
        dZ1 = (1 / batch_size) * (self.W2.T.dot(dZ2) * self.der_ReLU(Z1)) # (15 x 10) (10 x 41000) * elementwise (1 or 0) Back propogation eq. #2
        self.dW1 =  dZ1.dot(X.T) # (15 x 41000) (41000 x 784) -> (15 x 784) - - - Back propogation eq. #4
        self.db1 = np.sum(dZ1, axis = 1) # Back propogation eq. #3 - we are summing over cols in the backprop
        self.db1 = self.db1.reshape(15, 1)

    def update_params(self, lr):
        self.W1 = self.W1 - lr * self.dW1
        self.b1 = self.b1 - lr * self.db1
        self.W2 = self.W2 - lr * self.dW2
        self.b2 = self.b2 - lr * self.db2

    def get_predictions(self, A2):
        return np.argmax(A2, 0) # Return argmax along the rows (each example)

    def get_accuracy(self, predictions, Y): # prections and Y should be the same size
        return np.sum(predictions == Y) / Y.size
    
    def stochastic_gradient_descent(self, X_train, Y_train, X_dev, Y_dev, epochs, batch_size, lr):
        # Calculate the number of examples
        num_examples = X_train.shape[1]
        for i in range(epochs):
            # Generate a random permutation of indices
            permuted_indices = np.random.permutation(num_examples)
            # Shuffle both X and Y using the same permutation of indices
            X_shuffled = X_train[:, permuted_indices]
            Y_shuffled = Y_train[permuted_indices]
            # Iterate over the shuffled data in batches
            for j in range(0, num_examples, batch_size):
                X_batch = X_shuffled[:, j:j + batch_size]
                Y_batch = Y_shuffled[j:j + batch_size]
                # Forward propagation
                Z1, A1, Z2, A2 = self.forward_prop(X_batch)
                # Back propagation
                self.back_prop(Z1, A1, A2, X_batch, Y_batch, batch_size)
                # Update weights
                self.update_params(lr)

            print("Epoch:", i)
            # Calculate accuracy using the entire dataset
            Z1, A1, Z2, A2 = self.forward_prop(X_dev)
            accuracy = self.get_accuracy(self.get_predictions(A2), Y_dev)
            print("Accuracy:", accuracy)
            self.accuracies.append(accuracy)

    def plot_accuracies(self):
        epochs = range(1, len(self.accuracies) + 1)
        plt.plot(epochs, self.accuracies, 'b-', label='Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()