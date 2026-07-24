import numpy as np

# CASE I: Labels / Targets are Numbers
softmax_outputs = np.array([
    [0.7, 0.1, 0.2],
    [0.1, 0.5, 0.4],
    [0.02, 0.9, 0.08]
])

class_targets = np.array([0, 1, 1])
print(softmax_outputs[[0, 1, 2], class_targets])
print("="*100)

print(-np.log(softmax_outputs[range(len(softmax_outputs)), class_targets]))
print("="*100)

neg_log = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])
avg_loss = np.mean(neg_log)

print(avg_loss)
print("="*100)

# CASE II: Labels / Targets are One Hot Encoded

enc_class_targets = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0]
])

loss_matrix = softmax_outputs * enc_class_targets
loss_vector = np.sum(loss_matrix, axis=1)

print(f"Loss Vector: {loss_vector}")
print("="*100)

enc_neg_log = -np.log(loss_vector)
print(f"Negative Log: {enc_neg_log}")
print("="*100)

enc_avg_loss = np.mean(enc_neg_log)
print(f"Average Loss: {enc_avg_loss}")
print("="*100)

# main implementation

class Loss:
    def calculate(self, outputs, y):
        sample_losses = self.forward(outputs, y)
        mean_losses = np.mean(sample_losses)
        return mean_losses

class Loss_CategorialCrossEntropy(Loss):
    def forward(self, y_pred, y_test):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_test.shape) == 1:
            confidence = -np.log(y_pred_clipped[range(samples), y_test])
        elif len(y_test.shape) == 2:
            confidence = -np.log(np.sum(y_pred_clipped*y_test, axis=1))
        return confidence

calculate_loss = Loss_CategorialCrossEntropy()
mean_loss = calculate_loss.calculate(softmax_outputs, class_targets)
mean_loss_enc = calculate_loss.calculate(softmax_outputs, enc_class_targets)

print("="*100)
print(f"1-D Class Targets Losses:\n{mean_loss}")
print("="*100)
print(f"2-D Class Targets Losses:\n{mean_loss_enc}")
print("="*100)