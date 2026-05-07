inputs = [1, 2, 3, 2.5] # 4 inputs
weights = [             # 3 neurons * 4 inputs = 12 weights
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

bias = [2, 3, 0.5] # independent of inputs, can be any random value

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

# using loops to loop repeated steps
outputs = []

for n_weights, n_bias in zip(weights, bias): #for each list of lists of inputs
    n_output = 0 # temp variable to add o/p value
    for n_input, weight in zip(inputs, n_weights): # for each value of input inside list of lists.
        n_output += n_input*weight #add each x1.w11 + x1.w21+...
    n_output += n_bias #add respective bias

    outputs.append(n_output) #append output values in list.

print(f"Outputs:\ny1: {outputs[0]}\ny2: {outputs[1]}\ny3: {outputs[2]}")
print("_"*100)
