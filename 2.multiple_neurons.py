inputs = [1, 2, 3, 2.5] # 4 inputs
weights = [             # 3 neurons * 4 inputs = 12 weights
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

bias = [2, 3, 0.5] # independent of inputs, can be any random value

outputs = []

for n_weights, n_bias in zip(weights, bias):
    n_output = 0
    for n_input, n_weight in zip(inputs, n_weights):
        n_output += n_input*n_weight
    n_output += n_bias

    outputs.append(n_output)

print("Neuron Outputs: ")
for x in range(len(outputs)):
    print(f"y{x+1} = {outputs[x]}")


# outputs = [      # actual logic
#     #Neuron 1 (y1):
#     inputs[0] * weights[0][0] +
#     inputs[1] * weights[0][1] +
#     inputs[2] * weights[0][2] + bias[0],

#     #Neuron 2 (y2):
#     inputs[0] * weights[1][0] +
#     inputs[1] * weights[1][1] +
#     inputs[2] * weights[1][2] + bias[1],

#     #Neuron 1 (y1):
#     inputs[0] * weights[2][0] +
#     inputs[1] * weights[2][1] +
#     inputs[2] * weights[2][2] + bias[2]
# ]


