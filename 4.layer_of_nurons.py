import numpy as np

inputs = [
    [1.0, 2.0, 3.0, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

weights1 = [
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

bias1 = [2.0, 3.0, 0.5]

weights2 = [
    [0.1, -0.14, 0.5],
    [-0.5, 0.12, -0.33],
    [-0.44, 0.73, -0.13]
]

bias2 = [-1.0, 2.0, -0.5]

x = np.array(inputs)
v = np.array(weights1)
b1 = np.array(bias1)
w = np.array(weights2)
b2 = np.array(bias2)

z = np.dot(x, v.T) + b1
y = np.dot(z, w.T) + b2

print(f"Final output of Layered Neurons: ")
print(y)