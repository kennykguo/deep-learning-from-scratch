"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feed forward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
import random
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""

        # The length of the list is the number of layers
        self.num_layers = len(sizes) 
        # Size list type
        self.sizes = sizes 
        # Using list comprehension, we generate random values from the normal distribution, and append them to the biases and weights
        # random.randn(d0, d1, ..., dn) (for d as a dimension)
        self.biases = [] # self.biases is a list of numpy arrays
        for y in sizes[1:]:
            self.biases.append(np.random.randn(y, 1)) #creates a bias vector, of the shape (y,1)
        print("These are the randomized biases")
        print(self.biases)
        self.weights = []
        for x, y in zip(sizes[:-1], sizes[1:]):
            # We are flipping x and y in the rand function, to transpose the matrix
            self.weights.append(np.random.randn(y, x))
        print("These are the randomized weights")
        print(self.weights)
        # zip creates an iterable, that can be iterated over through list comprehension. The zip function combines multiple iterables into tuples
        # We are sure that this code creates the correct matrix dimensions for dot products
        # self.weights is a list of numpy arrays

    def feed_forward(self, a):
        """Return the output of the network if ``a`` is input."""
        # the zip function creates pairs of biases and weights that correspond to eachother for feedforward
        for b, w in zip(self.biases, self.weights): 
            # Sigmoid automatically applied element-wise to all data points
            a = sigmoid(np.dot(w, a)+b)
        #Return 10x1 numpy array
        # a = softmax(a)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None, save_file = None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        # Calculate the number of test samples
        test_data = list(test_data)
        n_test = len(test_data)  
        print("Number of test examples is: ")
        print(n_test)
        # Convert training_data to a list
        training_data = list(training_data)  
        # Calculate the number of training samples
        n = len(training_data)  
        print("Number of training examples is: ")
        print(n)

        # Loop through the specified number of epochs
        # For each epoch, we create mini-batches using the same training set, and only print out the evaluation after the entire training set has been iterated over
        for j in range(epochs):
            # Shuffle the training data to ensure randomness in mini-batches
            random.shuffle(training_data)
            # Split the training data into mini-batches of the specified size
            # Mini_batches is a list of training example batches
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            # Update the neural network parameters using each mini-batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            print("Epoch {0}: {1} / {2}".format(
                j, self.evaluate(test_data), n_test))
            
        if save_file:
            self.save_parameters(save_file)
            print("Weights and biases saved to:", save_file)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # print("This is what a mini batch looks like: ")
        # print(mini_batch)
        for x, y in mini_batch:
            # Run back propogation algorithm returns a list of a matrix to update 
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # Add the values matrix-wise to the list of matrices of change in b, and change in w
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # Update each of the weights matrices
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Feed forward again
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        # The first batch of activations, is the actual input
        zs = [] # list to store all the z vectors, layer by layer (z sigmoid)
        for b, w in zip(self.biases, self.weights):
            # Get weighted sum
            z = np.dot(w, activation)+b
            # Add weighted zum to the list of weighted sums
            zs.append(z)
            # Apply activation function
            activation = sigmoid(z)
            # Add the activation to the activation list
            activations.append(activation)
        
        # Backward pass
        # Back propogation algorithm # 1 (last activation)
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        #Back propogation algorithm #3 and #4
        nabla_b[-1] = delta
        # We are taking a dot product here. We can simply transpose a matrix, then do a dot product to get the Hadamard product.
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.


        for l in range(2, self.num_layers): # 3 layers - iterate over hidden layers
            z = zs[-l] # Weighted input for the current layer
            sp = sigmoid_prime(z) # Derivative of the activation function at the current layer
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp # Compute the error for the current layer
            nabla_b[-l] = delta  # Update the bias for the current layer
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) # Update the weight gradients for the current layer
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feed_forward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives partial C_x 
        and partial a for the output activations."""
        return (output_activations-y)
    
    def save_parameters(self, save_file):
        """Save the weights and biases to a text file."""
        with open(save_file, 'w') as f:
            f.write("Biases:\n")
            for b in self.biases:
                for val in np.nditer(b):
                    f.write(str(val) + "\n")
                f.write("\n")
            
            f.write("Weights:\n")
            for w in self.weights:
                for row in w:
                    for val in np.nditer(row):
                        f.write(str(val) + "\n")
                f.write("\n")

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))
    # return np.maximum(z, 0)

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
    # return z>0

# def softmax(z):
#     A = np.exp(z) / sum(np.exp(z))
#     return z