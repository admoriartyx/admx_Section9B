# This is the .py file for Task 1 of Section 9B

# Part a was completed by hand and a screenshot of the network is provided in the directory.

# Part b

import numpy as np

class MLP:
    def __init__(self, layer_sizes):
        self.layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i]) for i in range(self.layers-1)]
        self.biases = [np.zeros((1, size)) for size in layer_sizes[1:]]

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        activations = [x]
        for i in range(self.layers-1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            if i < self.layers-2:
                a = self.relu(z)
            else: 
                a = self.sigmoid(z)
            activations.append(a)
        return activations

    def backward(self, activations, y):
        deltas = [None] * self.layers
        deltas[-1] = activations[-1] - y
        
        for i in reversed(range(1, self.layers-1)):
            deltas[i] = np.dot(deltas[i+1], self.weights[i].T) * (activations[i] > 0)
        
        grad_w = [None] * (self.layers-1)
        grad_b = [None] * (self.layers-1)
        for i in range(self.layers-1):
            grad_w[i] = np.einsum('bi,bj->ij', activations[i], deltas[i+1])
            grad_b[i] = np.sum(deltas[i+1], axis=0, keepdims=True)
            self.weights[i] -= 0.01 * grad_w[i] 
            self.biases[i] -= 0.01 * grad_b[i]

