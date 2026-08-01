import numpy as np

weights = np.array([-3.0, -1.0, 2.0])
bias = 1.0
inputs = np.array([1.0, -2.0, 3.0])
target_output = 0.0
learning_rate = 0.001

def ReLU(x):
    return np.maximum(0, x)

def d_ReLU(x):
    return np.where(x > 0, 1.0, 0.0)

for iterations in range(200):
    linear_output = np.dot(weights, inputs) + bias
    output = ReLU(linear_output)
    loss = (output - target_output) ** 2

    dloss_doutput = 2 * (output - target_output)
    doutput_dlinear = d_ReLU(linear_output)
    dlinear_dweights = inputs
    dlinear_dbias = 1.0

    dloss_dlinear = dloss_doutput * doutput_dlinear
    dloss_dweights = dloss_dlinear * dlinear_dweights
    dloss_dbias = dloss_dlinear * dlinear_dbias

    weights -= learning_rate * dloss_dweights
    bias -= learning_rate * dloss_dbias

    print(f"Iteration {iterations + 1} Loss: {loss}")

print("="*100)
print(f"Final Weights: {weights}")
print(f"Final Biases: {bias}")