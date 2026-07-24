import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

# creating a layer of neurons
class LayerDense:
    def __init__(self, inputs, neurons):
        self.weights = 0.01 * np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))

    def forward(self, inputs):
        self.outputs = np.dot(inputs, self.weights) + self.biases

# ReLU activation
class ReLU:
    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)

# Softmax Activation
class Softmax:
    def forward(self, inputs):
        exp_vals = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.outputs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

# Loss Class
class Loss:
    def calculate(self, outputs, y):
        sample_losses = self.forward(outputs, y)
        mean_losses = np.mean(sample_losses)
        return mean_losses

# CCE Loss function
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_test):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_test.shape) == 1:
            confidence = -np.log(y_pred_clipped[range(samples), y_test])
        elif len(y_test.shape) == 2:
            confidence = -np.log(np.sum(y_pred_clipped*y_test, axis=1))
        return confidence

# importing data
x,y = spiral_data(samples=5, classes=3)

# creating instances for designing the network
l1 = LayerDense(2,3)
act1 = ReLU()
l2 = LayerDense(3,3)
act2 = ReLU()
l3 = LayerDense(3,3)
act3 = Softmax()

# Building the Neural Network
# Layer 1 ReLU
l1.forward(x)
act1.forward(l1.outputs)

# Layer 2 ReLU
l2.forward(act1.outputs)
act2.forward(l2.outputs)

# Layer 3 Softmax
l3.forward(act2.outputs)
act3.forward(l3.outputs)

softmax_outputs = act3.outputs

print(f"Softmax Outputs:\n{softmax_outputs[:5]}")
print("="*100)

ccel = Loss_CategoricalCrossEntropy()
mean_loss = ccel.calculate(softmax_outputs, y)

predictions = np.argmax(softmax_outputs, axis=1)
if len(y.shape) == 2:
    y_true = np.argmax(y, axis=1)
else:
    y_true = y
accuracy = np.mean(predictions == y_true)

print(f"Mean Loss: {mean_loss}")
print("="*100)

print(f"Accuracy: {accuracy}")
print("="*100)