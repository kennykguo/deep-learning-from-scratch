import numpy as np
import scipy as signal
import math

class ConvolutionalNN:
    def __init__(self):
        # Convolution
        self.layer_weights = np.random.randn(5, 5, 2) * math.sqrt(2. / 5)
        self.delta_conv_weights = np.zeros_like(self.layer_weights)
        self.layer_bias = np.zeros((24, 24, 2))
        self.delta_conv_bias = np.zeros_like(self.layer_bias)
        
        # Fully connected
        self.fc_weights = np.random.randn(10, 288) * math.sqrt(2. / 288)
        self.delta_fc_weights = np.zeros_like(self.fc_weights)
        self.fc_bias = np.zeros((10, 1))
        self.delta_fc_bias = np.zeros_like(self.fc_bias)
        self.delta_fc_bias = self.delta_fc_bias.reshape(10, 1)

        # Batch normalization
        self.gamma_conv = np.ones((24, 24, 2))
        self.delta_gamma_conv = np.zeros_like(self.gamma_conv)
        self.beta_conv = np.zeros((24, 24, 2))
        self.delta_beta_conv = np.zeros_like(self.delta_beta_conv)
        self.gamma_fc = np.ones((10, 1))
        self.delta_gamma_fc = np.zeros_like(self.gamma_fc)
        self.beta_fc = np.zeros((10, 1))
        self.delta_beta_fc = np.zeros_like(self.beta_fc)



    def max_pooling(input_data):
        # (24, 24, 2)
        input_height, input_width, input_depth = input_data.shape

        # Calculate the output dimensions
        output_height = input_height // 2 # 12
        output_width = input_width // 2 # 12
        output_depth = input_depth # 2 - depth stays the same

        # Initialize the output array and array to store indices
        output_data = np.zeros((output_height, output_width, output_depth))
        indices = np.zeros((output_height, output_width, output_depth, 2), dtype=int)

        # Apply max pooling
        for h in range(output_height):
            for w in range(output_width):
                for d in range(output_depth):
                    # Extract the 2x2 region of interest from the input data
                    region = input_data[h*2:(h+1)*2, w*2:(w+1)*2, d]
                    # Compute the maximum value in the region
                    max_val = np.max(region)
                    output_data[h, w, d] = max_val
                    # Find the indices of the maximum value in the region
                    max_indices = np.unravel_index(np.argmax(region), region.shape)
                    # Store the indices relative to the region and convert to global indices
                    indices[h, w, d] = [h*2 + max_indices[0], w*2 + max_indices[1]]

        return output_data, indices



    def forward_propagation(self, layer_input, dropout_rate):
        # Convolution
        for i in range(2): # 2 filters in total
            layer_output[:,:,i] = signal.correlate2d(layer_input, self.layer_weights[:,:,i], mode='valid')
        layer_output = layer_output + self.layer_bias   # (24, 24, 2)
        
        # Batch Normalization for Convolutional Layer
        layer_output, bn_cache_conv = self.batch_norm_forward(layer_output, self.gamma_conv, self.beta_conv)
        
        # Activation layer
        layer_output = self.ReLU(layer_output)  # (24, 24, 2)
        
        # Pool layer
        layer_pool, layer_indices = self.max_pooling(layer_output)  # (12, 12, 2)
        
        # Flattening
        self.layer_pool = layer_pool.reshape(288, 1) # (288, 1)
        
        # Dropout
        dropout_mask = (np.random.rand(*layer_pool.shape) < dropout_rate) / dropout_rate
        layer_pool *= dropout_mask # (288, 1)
        
        # Fully connected layer
        final_output = self.fc_weights.dot(layer_pool)  # (10, 288) (288, 1) = (10, 1)
        final_output = final_output + self.fc_bias # (10, 1) + (10, 1) = (10, 1)
        
        # Batch Normalization for Fully Connected Layer
        final_output, bn_cache_fc = self.batch_norm_forward(final_output, self.gamma_fc, self.beta_fc) # (10, 1)
        
        final_output = self.softmax(final_output) # (10, 1)
        return layer_output, layer_pool, layer_indices, final_output, bn_cache_conv, bn_cache_fc

    def back_prop(self, layer_input, layer_output, layer_pool, layer_indices, final_output, label,  bn_cache_conv, bn_cache_fc, dropout_mask):
        # Backpropagate cost
        x = self.create(label)
        dZ = (final_output - x)  # (10, 1) - (10, 1) = (10, 1)
        
        # Backpropagate through Batch Normalization for Fully Connected Layer
        dZ, dgamma_fc, dbeta_fc = self.batch_norm_backward(dZ, bn_cache_fc)
        
        # Backpropagate weights and biases for Fully Connected Layer
        delta_fc_weights = dZ.dot(layer_pool.T)  # (10, 1) (1, 288) = (10, 288)
        delta_fc_bias = dZ
        
        # Backpropagate error
        dZ_pool_output = np.dot(self.fc_weights.T, dZ) * self.der_ReLU(layer_pool)  # (288, 10) (10, 1) = (288, 1)
        
        # Undo Dropout
        dZ_pool_output *= dropout_mask
        
        # Unflattening
        dZ_pool_output = dZ_pool_output.reshape(12, 12, 2)
        
        # Unpooling
        dZ_pool_input = np.zeros((24, 24, 2))
        for i in range(12):  # height
            for j in range(12):  # width
                for k in range(2):  # depth
                    # Get the global indices from layer_indices
                    x_index, y_index = layer_indices[i, j, k]
                    # Assign the gradient from dZ_pool_output to the corresponding position in dZ_pool_input
                    dZ_pool_input[x_index, y_index, k] = dZ_pool_output[i, j, k]
        
        # Backpropagate through ReLU activation
        dZ_pool_input *= self.der_ReLU(layer_output)
        
        # Backpropagate through Batch Normalization
        dZ_pool_input, dgamma_conv, dbeta_conv = self.batch_norm_backward(dZ_pool_input, bn_cache_conv)
        
        # Backpropagate Conv layer
        for i in range(2):  # For each filter in the kernel
            self.delta_conv_weights[:, :, i] = signal.correlate(layer_input, dZ_pool_input[:, :, i], mode="valid")
        self.delta_conv_bias = dZ_pool_input
        
    def update_params(self, learning_rate):
        layer_weights -= learning_rate * self.delta_conv_weights
        layer_bias -= learning_rate * self.delta_conv_bias
        fc_weights -= learning_rate * self.delta_fc_weights
        fc_bias -= learning_rate * self.delta_fc_bias
        gamma_conv -= learning_rate * self.delta_gamma_conv
        beta_conv -= learning_rate * self.delta_beta_conv
        gamma_fc -= learning_rate * self.delta_gamma_fc
        beta_fc -= learning_rate * self.delta_beta_fc


    # Helper functions (only using ReLU but can use others)
    def get_prediction(A2):
        return np.argmax(A2, 0)
    
    def create(Y):
        column_Y = np.zeros((10, 1))
        column_Y[Y] = 1
        column_Y = column_Y.T
        return column_Y.reshape(10,1)
    
    def der_ReLU(Z):
        return Z > 0
    
    def ReLU2(Z, alpha=0.01):
        return np.where(Z > 0, Z, alpha * Z)
    
    def ReLU(Z):
        return np.maximum(Z, 0)
    
    def ReLU2(Z):
        return np.maximum(Z, 0)
    
    def sigmoid(z):
        # Compute the sigmoid function element-wise
        return 1.0 / (1.0 + np.exp(-z))
    
    def sigmoid_prime(self,z):
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    def softmax(Z):
        # Apply softmax column-wise
        exp_Z = np.exp(Z - np.max(Z, axis=0)) # Subtracting the maximum value in each column to avoid overflow
        return exp_Z / np.sum(exp_Z, axis=0)