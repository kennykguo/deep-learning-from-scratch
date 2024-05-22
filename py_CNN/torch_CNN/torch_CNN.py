import torch
import scipy as signal

class Conv2d:
    def __init__(self, input_size, in_channels, out_channels, kernel_size, batch_size, bias = True):
        # in_channels is the depth of the input
        # out_channels is the number of filters
        # We will fix stride to 1, and padding to 0
        # Assume the input, output and kernel are squares
        self.input_size = input_size # 28
        self.in_channels = in_channels # 3

        self.output_size = self.input_size - self.kernel_size + 1 # 24
        self.out_channels = out_channels # 12

        self.kernel_size = kernel_size # 5
        self.stride = 1
        self.batch_size = batch_size
        self.filters = []

        # Create a total of out_channel filters (12)
        for _ in range(out_channels):
            self.filters.append(torch.randn((
                self.kernel_size, self.kernel_size, in_channels
                )))
        # Create the bias (1)
        self.bias = torch.randn((self.output, self.output, 1))
        

    def __call__(self, x):
        # Input size eg.
        # (28, 28, 3, 32)
        # Determine the size of the output eg.
        # eg. (24, 24, 12, 32)
        self.output = torch.zeros((self.output_size, self.output_size, self.out_channels, x.shape[3]))
        # Loop through each example in the batch
        for example in range(x.shape[3]):
            # Loop through each filter
            for filter in range(self.out_channels):
                # Loop through height and width
                for i in range (self.output_size):
                    for j in range (self.output_size):
                        # (5, 5, 3, 1) eg.
                        input_patch = x[i:self.kernel_size + i, j:self.kernel_size + j, :, example].reshape((self.kernel_size, self.kernel_size, -1))
                        # Should be (5, 5, 3) eg.
                        correlation = self.filters[filter]
                        self.output[i, j, filter, example] = torch.sum(input_patch * correlation)
        self.output += self.bias
        return self.output



class Linear:
    def __init__(self, fan_in, fan_out, bias = True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in ** 0.5
        self.bias = torch.zeros(fan_out) if bias else None

    # The class can be called like a function, which takes in the input x, and returns the respective output
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])



class Tanh:
  def __call__(self, x):
    self.out = torch.tanh(x)
    return self.out

  def parameters(self):
    return []
  


class Sequential:
    def __init__(self, layers):
        self.layers = layers
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        # Get all parameters and put them in a list
        return [p for layer in self.layers for p in layer.parameters()]