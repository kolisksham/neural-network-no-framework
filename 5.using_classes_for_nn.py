import numpy as np
import nnfs #dataset 
from nnfs.datasets import spiral_data
nnfs.init()

class Layer_Dense: # class whose methods can be called again and again without writing codes repeatedly
    def __init__(self, inputs, neurons):
        # creating non-linear dataset
        self.weights = 0.01 * np.random.randn(inputs, neurons) # creates a random matrix of shape (inputs X neurons) multiplying with 0.01 gives very small numbers output
        self.biases = np.zeros((1, neurons)) # bias values 0, vector of shape (1 X no. of neurons)

    def forward_pass(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
    
x, y = spiral_data(samples=100, classes=3) # 100 lines of data, 3 different classes, 300 total values

dense1 = Layer_Dense(2, 3) # 2 inputs 3 neurons
dense1.forward_pass(x)

print(dense1.output[:5])