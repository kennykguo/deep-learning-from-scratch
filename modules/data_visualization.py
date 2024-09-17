"""
This module provides functions for visualizing various aspects of neural network training, including activations, gradients, and weight updates. Each function generates histograms and plots to assist in diagnosing issues and monitoring training progress.

Functions:
1. `plot_activation_distribution(layers)`: Visualizes the distribution of activations for each layer (excluding the output layer).
2. `plot_gradient_distribution(layers)`: Visualizes the distribution of gradients for each layer (excluding the output layer).
3. `plot_weights_gradient_distribution(parameters)`: Visualizes the distribution of weight gradients.
4. `plot_weight_update_ratios(parameters, updates_data)`: Plots the ratio of actual updates to data over time to assess if the learning rate is appropriate.

Usage:
- Ensure `plt` (matplotlib) and `torch` are imported.
- Provide appropriate `layers`, `parameters`, and `updates_data` as inputs to the functions.
- Call the functions to generate the respective plots and histograms.
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def plot_activation_distribution(layers):
    """
    Visualizes the distribution of activations for each layer (excluding the output layer).
    
    Args:
    layers (list): A list of neural network layers.

    Returns:
    None
    """
    plt.figure(figsize=(20, 4))  # Width and height of the plot
    legends = []
    for i, layer in enumerate(layers[:-1]):  # Note: exclude the output layer
        if isinstance(layer, nn.Tanh):
            t = layer.out
            print(f'Layer {i} ({layer.__class__.__name__}): Mean {t.mean():+.2f}, Std {t.std():.2f}, Saturated: {(t.abs() > 0.97).float().mean() * 100:.2f}%')
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f'Layer {i} ({layer.__class__.__name__})')
    plt.legend(legends)
    plt.title('Activation Distribution')
    plt.show()

def plot_gradient_distribution(layers):
    """
    Visualizes the distribution of gradients for each layer (excluding the output layer).
    
    Args:
    layers (list): A list of neural network layers.

    Returns:
    None
    """
    plt.figure(figsize=(20, 4))  # Width and height of the plot
    legends = []
    for i, layer in enumerate(layers[:-1]):  # Note: exclude the output layer
        if isinstance(layer, nn.Tanh):
            t = layer.out.grad
            print(f'Layer {i} ({layer.__class__.__name__}): Mean {t.mean():+f}, Std {t.std():e}')
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f'Layer {i} ({layer.__class__.__name__})')
    plt.legend(legends)
    plt.title('Gradient Distribution')
    plt.show()

def plot_weights_gradient_distribution(parameters):
    """
    Visualizes the distribution of weight gradients.
    
    Args:
    parameters (list): A list of parameters (weights and biases) of the model.

    Returns:
    None
    """
    plt.figure(figsize=(20, 4))  # Width and height of the plot
    legends = []
    for i, p in enumerate(parameters):
        t = p.grad
        if p.ndim == 2:
            print(f'Weight {tuple(p.shape)} | Mean {t.mean():+f} | Std {t.std():e} | Grad:data ratio {t.std() / p.std():e}')
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f'{i} {tuple(p.shape)}')
    plt.legend(legends)
    plt.title('Weights Gradient Distribution')
    plt.show()

def plot_weight_update_ratios(parameters, updates_data):
    """
    Plots the ratio of actual updates to data over time to assess if the learning rate is appropriate.
    
    Args:
    parameters (list): A list of parameters (weights and biases) of the model.
    updates_data (list of lists): A list where each sublist contains update values for each parameter over time.

    Returns:
    None
    """
    plt.figure(figsize=(20, 4))
    legends = []
    for i, p in enumerate(parameters):
        if p.ndim == 2:
            plt.plot([update[i] for update in updates_data])
            legends.append(f'Param {i}')
    plt.plot([0, len(updates_data)], [-3, -3], 'k')  # These ratios should be ~1e-3; -3 from log scale is a rough heuristic
    plt.legend(legends)
    plt.title('Weight Update Ratios')
    plt.show()
