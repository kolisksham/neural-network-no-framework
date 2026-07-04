import numpy as np

# single neuron
inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2

outputs = np.dot(weights, inputs) + bias

print("="*100)
print("Single Neuron Outputs: ")
print(f"y = {outputs}")
print("="*100)

# multiple neurons
inputs = [1, 2, 3, 2.5]
weights = [
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

bias = [2, 3, 0.5]

outputs = np.dot(weights, inputs) + bias

print("Layered Neurons Outputs: ")
for x in range(len(outputs)):
    print(f"y{x+1} = {outputs[x]}")
print("="*100)

# batch of inputs in multiple neurons
inputs = [
    [1.0, 2.0, 3.0, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

weights = [
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

bias = [2.0, 3.0, 0.5]

outputs = np.dot(inputs, np.array(weights).T + bias)

print("Batch of Input Data Outputs: ")
print(outputs)