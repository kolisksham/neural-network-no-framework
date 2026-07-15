import numpy as np

# CASE I: Labels / Targets are Numbers
softmax_outputs = np.array([
    [0.7, 0.1, 0.2],
    [0.1, 0.9, 0.4],
    [0.02, 0.9, 0.08]
])

class_targets = [0, 1, 1]
print(softmax_outputs[[0, 1, 2], class_targets])
print("="*100)

print(-np.log(softmax_outputs[range(len(softmax_outputs)), class_targets]))
print("="*100)

neg_log = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])
avg_loss = np.mean(neg_log)

print(avg_loss)
print("="*100)

# CASE II: Labels / Targets are One Hot Encoded
