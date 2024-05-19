import numpy as np

class Network:
    def __init__(self, batch_size, lr):
        # Parameters
        self.W1 = np.random.rand(15, 784) - 0.5
        self.W1 *= 0.1;
        self.b1 = np.random.zeros(15, batch_size)
        self.W2 = np.random.rand(10, 15)
        self.b2 = np.random.rand(10, batch_size)
        # Gradients
        self.dW1 = np.random.zeros(15, 784)
        self.db1 = np.random.zeros(15, batch_size)
        self.dW2 = np.random.zeros(10, 15)
        self.db2 = np.random.zeros(10, batch_size)
        self.lr = lr
        self.batch_size = batch_size
    
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
        Z1 = self.W1.dot(X) + self.b1 # (10 x 784) (784 x n) + (10 x n) -> (10 x n)
        A1 = self.ReLU(Z1)
        Z2 = self.W2.dot(A1) + self.b2 # (10 x 10) (10 x n) + (10 x n) = (10 x n)
        A2 = self.softmax(Z2)
        return Z1, A1, Z2, A2

    def back_prop(self, Z1, A1, A2, W2, X, Y):
        mat_Y = self.create(Y)
        dZ2 = (1 / self.batch_size) * (A2 - mat_Y) # 10 x 41000 - - - Back propogation eq. #1
        self.dW2 = dZ2.dot(A1.T) # (10 x 41000) (41000 x 10) -> (10 x 10) - - - Back propogation eq. #4
        self.db2 = np.sum(dZ2, axis = 1) # Back propogation eq. #3
        dZ1 = (1 / self.batch_size) * (W2.T.dot(dZ2) * self.der_ReLU(Z1)) # (10 x 10) (10 x 41000) * elementwise (1 or 0) Back propogation eq. #2
        self.dW1=  dZ1.dot(X.T) # (10 x 41000) (41000 x 784) -> (10 x 784) - - - Back propogation eq. #4
        self.db1 = np.sum(dZ2, axis = 1) # Back propogation eq. #3

    def update_params(self):
        self.W1 = self.W1 - self.lr * self.dW1
        self.b1 = self.b1 - self.lr * self.db1
        self.W2 = self.W2 - self.lr * self.dW2
        self.b2 = self.b2 - self.lr * self.db2


    def get_predictions(self, A2):
        return np.argmax(A2, 0)

    def get_accuracy(self, predictions, Y):
        # print(predictions)
        # print(Y)
        return np.sum(predictions == Y) / Y.size
    
    def stochastic_gradient_descent(self, X, Y, epochs, alpha, batch_size):
        # Calculate the number of examples
        num_examples = X.shape[1]
        for i in range(epochs):

            # Generate a random permutation of indices
            permuted_indices = np.random.permutation(num_examples)
            # Shuffle both X and Y using the same permutation of indices
            X_shuffled = X[:, permuted_indices]
            Y_shuffled = Y[permuted_indices]

            # Iterate over the shuffled data in batches
            for j in range(0, num_examples, batch_size):
                X_batch = X_shuffled[:, j:j + batch_size]
                Y_batch = Y_shuffled[j:j + batch_size]
                # Forward propagation
                Z1, A1, Z2, A2 = self.forward_prop(W1, b1, W2, b2, X_batch, batch_size)
                # Back propagation
                dW1, db1, dW2, db2 = self.back_prop(Z1, A1, Z2, A2, W1, W2, X_batch, Y_batch, batch_size)
                # Update weights
                W1, b1, W2, b2 = self.update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

            print("Epoch:", i)
            # Calculate accuracy using the entire dataset
            Z1, A1, Z2, A2 = self.forward_prop(W1, b1, W2, b2, X_batch, batch_size)
            print("Accuracy:", self.get_accuracy(self.get_predictions(A2), Y_batch))