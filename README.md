# Neural Network from Scratch

This repository is a beginner-friendly introduction to neural networks built entirely with Python and NumPy. The scripts focus on the core ideas behind neural networks without relying on a high-level framework.

## What this project covers

These examples walk through the building blocks of neural networks:

- A single neuron and how it computes an output
- Multiple neurons working together
- Vectorized math using NumPy
- Layers of neurons
- Reusable class-based layers
- ReLU and Softmax activation functions
- A small complete neural network example

## Project structure

- 1.neuron.py - Demonstrates a single neuron with manual computation.
- 2.multiple_neurons.py - Shows how multiple neurons are computed in parallel.
- 3.using_numpy_in_nn.py - Uses NumPy for efficient matrix-based neural network calculations.
- 4.layer_of_neurons.py - Builds a simple layer-based neural network example.
- 5.using_classes_for_nn.py - Implements a reusable dense layer using Python classes.
- 6.ReLU_and_Softmax.py - Demonstrates ReLU and Softmax activation functions.
- 7.full_neural_network_no_loss.py - Shows a small neural network forward pass using layers and activations.
- basics/ - Extra NumPy examples for broadcasting and summation.
- data/ - Sample datasets used for learning and experimentation.

## Requirements

Make sure Python is installed, then install the required packages:

`ash
pip install numpy nnfs
`

## How to run

Run any script with Python, for example:

`ash
python 1.neuron.py
`

You can also try:

`ash
python 3.using_numpy_in_nn.py
python 6.ReLU_and_Softmax.py
python 7.full_neural_network_no_loss.py
`

## Learning goals

This project is intended for learners who want to understand:

- how neurons compute outputs
- how weights and biases influence results
- how NumPy simplifies neural network math
- how layers and activation functions fit together
- how a basic network produces predictions

## Notes

These examples are intentionally simple and educational. They are designed to build intuition rather than serve as a production-ready neural network implementation.
