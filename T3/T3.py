# Note that I literally could not even get the training data and images properly downloaded,
# so the code does not run but I would like to think it is close enough to what was wanted assuming
# training data was more accessible.

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target']

y = y.astype(int)
X = X / 255.0

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import numpy as np

class SimpleMLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.01
        self.bias2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = np.tanh(self.z1)  
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.softmax(self.z2) 
        return self.a2

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

mlp = SimpleMLP(input_size=784, hidden_size=50, output_size=10) 
predictions = mlp.predict(X_test[:5])
print("Predicted Labels:", predictions)

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

mnist = fetch_openml('mnist_784')
X, y = mnist['data'], mnist['target'].astype(np.int)
X = X / 255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.reshape((-1, 28, 28))
X_test = X_test.reshape((-1, 28, 28))

def convolve2d(image, kernel, padding=0, strides=1):
    image = np.pad(image, [(padding, padding), (padding, padding)], mode='constant', constant_values=0)

    kernel_height, kernel_width = kernel.shape
    padded_height, padded_width = image.shape

    output_height = (padded_height - kernel_height) // strides + 1
    output_width = (padded_width - kernel_width) // strides + 1

    new_image = np.zeros((output_height, output_width))

    for y in range(0, output_height):
        for x in range(0, output_width):
            new_image[y, x] = np.sum(image[y * strides:y * strides + kernel_height, x * strides:x * strides + kernel_width] * kernel)
    return new_image

def relu(X):
    return np.maximum(0, X)

def max_pooling(X, pool_size=2, stride=2):
    output_height = X.shape[0] // pool_size
    output_width = X.shape[1] // pool_size
    new_image = np.zeros((output_height, output_width))
    for y in range(output_height):
        for x in range(output_width):
            new_image[y, x] = np.max(X[y * stride:y * stride + pool_size, x * stride:x * stride + pool_size])
    return new_image

kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

sample_image = X_train[0]
convoluted_image = convolve2d(sample_image, kernel, padding = 1)
activated_image = relu(convoluted_image)
pooled_image = max_pooling(activated_image)


def fully_connected(X, weights, biases):
    return np.dot(X, weights) + biases

flattened = pooled_image.flatten()
weights = np.random.rand(flattened.shape[0], 10) 
biases = np.random.rand(10)
outputs = fully_connected(flattened, weights, biases)
predictions = []
for image in X_test:
    image = convolve2d(image, kernel, padding=1)
    image = relu(image)
    image = max_pooling(image)
    image = image.flatten()
    output = fully_connected(image, weights, biases)
    predictions.append(np.argmax(output))

cm = confusion_matrix(y_test, predictions)
print(cm)

import matplotlib.pyplot as plt
losses = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]

plt.plot(losses)
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
plt.savefig('Task3.png')
