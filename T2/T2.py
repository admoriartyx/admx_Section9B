# This is the .py file for Task 2 of Section 9B

# Part a was completed by hand and the image is located in the current directory.

# Part b

import numpy as np

def conv_forward(X, W, b, stride=1, padding=0):
    pass

def relu(X):
    return np.maximum(0, X)

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

class CNN:
    def __init__(self, input_dim, num_filters, filter_size, pool_size, hidden_dims, output_dim):
        self.params = {
            'W1': np.random.randn(num_filters, input_dim, filter_size, filter_size),
            'b1': np.zeros((num_filters, 1)),
            'W2': np.random.randn(hidden_dims, (input_dim // pool_size) * (input_dim // pool_size) * num_filters),
            'b2': np.zeros((hidden_dims, 1)),
            'W3': np.random.randn(output_dim, hidden_dims),
            'b3': np.zeros((output_dim, 1))
        }

    def forward(self, X):
        h1 = conv_forward(X, self.params['W1'], self.params['b1'])
        a1 = relu(h1)
        h2 = np.dot(a1.flatten(), self.params['W2']) + self.params['b2'].T
        a2 = relu(h2)
        out = np.dot(a2, self.params['W3']) + self.params['b3'].T
        output = sigmoid(out)
        
        return output

    def backward(self):
        pass

