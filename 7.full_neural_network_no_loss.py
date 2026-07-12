import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

# classes
class Layer_Dense:
    def __init__(self, inputs, neurons):
        self.weights = 0.01 * np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))

    def forward_pass(self, inputs):
        self.outputs = np.dot(inputs, self.weights) + self.biases
    
class ReLU:
    def forward_pass(self, inputs):
        self.outputs = np.maximum(0, inputs)
    
class Softmax:
    def forward_pass(self, inputs):
        exp_vals = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.outputs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

# main 

# inputs
x, y = spiral_data(samples=100, classes=3)

# creating instances
l1 = Layer_Dense(2, 3) # 2 inputs and 3 neurons in Layer 1
act1 = ReLU() # activation for Layer 1
l2 = Layer_Dense(3, 3) #3 inputs (outputs from previous layer) and 3 neurons in Layer 2
act2 = Softmax()

# creating neural network

# layer 1 ReLU Activation
l1.forward_pass(x)
act1.forward_pass(l1.outputs)

# layer 2 Softmax Activation
l2.forward_pass(act1.outputs)
act2.forward_pass(l2.outputs)

print("="*100)
print("Neural Network Output:")
print("="*100)
print(act2.outputs[:5])
print("="*100)