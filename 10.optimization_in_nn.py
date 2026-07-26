import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import vertical_data
nnfs.init()

x,y = vertical_data(samples=100, classes=3)
plt.scatter(x[:, 0], x[:,1], c=y, s=40, cmap='brg')
# plt.show()

class Layer_Dense:
    def __init__(self, inputs, neurons):
        self.weights = 0.01 * np.random.randn(inputs, neurons);
        self.biases = np.zeros((1, neurons))
    def forward(self, inputs):
        self.outputs = np.dot(inputs, self.weights) + self.biases

class ReLU:
    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)
    
class Softmax:
    def forward(self, inputs):
        exp_vals = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.outputs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

class Loss:
    def calculate(self, outputs, y):
        sample_losses = self.forward(outputs, y)
        mean_losses = np.mean(sample_losses)
        return mean_losses

class Loss_CCEL(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if y_true.ndim == 1:
            confidence = -np.log(y_pred_clipped[range(samples), y_true])
        elif y_true.ndim == 2:
            confidence = -np.log(np.sum(y_pred_clipped * y_true, axis=1))
        return confidence

# Strategy 1: Randomly Select weights/biases (DOES NOT WORK)

# # initiating instances
# l1 = Layer_Dense(2,3)
# act1 = ReLU()
# l2 = Layer_Dense(3,3)
# act2 = Softmax()
# cceloss = Loss_CCEL()

# # helper variables
# lowest_loss = 9999999
# best_l1_wgt = l1.weights.copy()
# best_l1_bias = l1.biases.copy()
# best_l2_wgt = l2.weights.copy()
# best_l2_bias = l2.biases.copy()

# for iteration in range(100000):
#     l1.weights = 0.05*np.random.randn(2,3)
#     l1.biases = 0.05*np.random.randn(1,3)
#     l2.weights = 0.05*np.random.randn(3,3)
#     l2.biases = 0.05*np.random.randn(1,3)

#     l1.forward(x)
#     act1.forward(l1.outputs)
#     l2.forward(act1.outputs)
#     act2.forward(l2.outputs)

#     loss = cceloss.calculate(act2.outputs, y)

#     predictions = np.argmax(act2.outputs, axis=1)
#     if y.ndim == 2:
#         y_true = np.argmax(y, axis=1)
#     else:
#         y_true = y

#     accuracy = np.mean(predictions == y)

#     if loss < lowest_loss:
#         print(f"New set of parameters found. Iteration: {iteration} Loss: {loss} Accuracy: {accuracy}")
#         best_l1_wgt = l1.weights.copy()
#         best_l1_bias = l1.biases.copy()
#         best_l2_wgt = l2.weights.copy()
#         best_l2_bias = l2.biases.copy()
#         lowest_loss = loss

# print("All iterations Done!")

# Strategy 2: Randomly Adjust weights/biases (WORKS BETTER BUT FAILS FOR COMPLEX DATA)

# initiating instances
l1 = Layer_Dense(2,3)
act1 = ReLU()
l2 = Layer_Dense(3,3)
act2 = Softmax()
cceloss = Loss_CCEL()

# helper variables
lowest_loss = 9999999
best_l1_wgt = l1.weights.copy()
best_l1_bias = l1.biases.copy()
best_l2_wgt = l2.weights.copy()
best_l2_bias = l2.biases.copy()

for iteration in range(100000):
    l1.weights += 0.05*np.random.randn(2,3)
    l1.biases += 0.05*np.random.randn(1,3)
    l2.weights += 0.05*np.random.randn(3,3)
    l2.biases += 0.05*np.random.randn(1,3)

    l1.forward(x)
    act1.forward(l1.outputs)
    l2.forward(act1.outputs)
    act2.forward(l2.outputs)

    loss = cceloss.calculate(act2.outputs, y)

    predictions = np.argmax(act2.outputs, axis=1)
    if y.ndim == 2:
        y_true = np.argmax(y, axis=1)
    else:
        y_true = y

    accuracy = np.mean(predictions == y)

    if loss < lowest_loss:
        print(f"New set of parameters found. Iteration: {iteration} Loss: {loss} Accuracy: {accuracy}")
        best_l1_wgt = l1.weights.copy()
        best_l1_bias = l1.biases.copy()
        best_l2_wgt = l2.weights.copy()
        best_l2_bias = l2.biases.copy()
        lowest_loss = loss

    else:
        l1.weights = best_l1_wgt.copy()
        l1.biases = best_l1_bias.copy()
        l2.weights = best_l2_wgt.copy()
        l2.biases = best_l2_bias.copy()

print("All iterations Done!")





