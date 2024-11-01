"""
This module provides basic neural network components:

1. **Linear**: A fully connected layer with optional bias. It applies the input matrix to weights and adds bias if specified.
2. **Tanh**: Applies the Tanh activation function element-wise.
3. **Sequential**: Chains layers together for forward passes.

**Usage**:
Create a model using `Sequential` with a list of layers and retrieve all parameters for optimization.
"""

import torch

class Linear:
    def __init__(self, fan_in, fan_out, bias = True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in ** 0.5
        self.bias = torch.zeros(fan_out) if bias else None

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
        params = []
        for layer in self.layers:
            params += layer.parameters()  # Collect parameters from each layer
        return params