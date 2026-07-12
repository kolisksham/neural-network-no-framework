import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

# ReLU
class ReLU:
    def forward_pass(self, inputs):
        self.outputs = np.maximum(0, inputs)

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
relu = ReLU()
relu.forward_pass(inputs)
print(f"ReLU activation Outputs:\n{relu.outputs}")
print("=" * 100)

# Softmax

class Softmax:
    def forward_pass(self, inputs):
        exp_vals = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.prob = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        self.outputs = self.prob

inputs = [
    [1.0, 2.0, 3.0, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

softm = Softmax()
softm.forward_pass(inputs)
print("Softmax Outputs:")
# print(softm.prob)
print(softm.outputs)
print("="*100)

# ReLU + Softmax

class Dense_Layer:
    def __init__(self, inputs, neurons):
        self.weights = 0.01*np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))

    def forward_pass(self, inputs):
        self.outputs = np.dot(inputs, self.weights) + self.biases

l1 = Dense_Layer(2, 3) # hidden layer (2 i/ps, 3 neuron)
act1 = ReLU()
l2 = Dense_Layer(3, 3) # output layer (3 i/ps, 3 o/ps)
act2 = Softmax()

# forward pass

x, y = spiral_data(samples= 100, classes= 3)

# hidden layer and it's activation
l1.forward_pass(x)
act1.forward_pass(l1.outputs)

# output layer and it's activation
l2.forward_pass(act1.outputs)
act2.forward_pass(l2.outputs)

outputs = act2.outputs

print("="*100)
print("Final Output (First 5)")
print("="*100)
print(outputs[:5])
print("="*100)
