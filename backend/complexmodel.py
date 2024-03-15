import torch.nn as nn
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

class ComplexModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=10, num_hidden_layers=10):
        super(ComplexModel, self).__init__()

        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        self.layers.extend([nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers - 1)])
        self.layers.append(nn.Linear(hidden_size, output_size))
        self.leaky_relu = nn.LeakyReLU()  # Use LeakyReLU activation function

    def forward(self, x):
        for layer in self.layers:
            x = self.leaky_relu(layer(x))
        return x