# Neural Network from Scratch

This repository is a beginner-friendly introduction to neural networks built with Python and NumPy. The examples focus on the core ideas behind neural networks without relying on a high-level framework.

## What this project covers

These scripts walk through the building blocks of a neural network:

- a single neuron and how it computes an output
- multiple neurons working together
- vectorized math with NumPy
- layers of neurons
- reusable class-based layers
- ReLU and Softmax activation functions
- a small complete forward pass through a neural network

## Project structure

- 1.neuron.py — demonstrates a single neuron with manual computation
- 2.multiple_neurons.py — shows how multiple neurons are computed in parallel
- 3.using_numpy_in_nn.py — uses NumPy for matrix-based neural network math
- 4.layer_of_neurons.py — builds a simple layer-based network example
- 5.using_classes_for_nn.py — implements a reusable dense layer with classes
- 6.ReLU_and_Softmax.py — demonstrates ReLU and Softmax activations
- 7.full_neural_network_no_loss.py — shows a small neural network forward pass
- basics/ — extra NumPy examples for broadcasting and summation
- data/ — sample datasets for learning and experimentation

## Requirements

Make sure Python is installed, then install the required packages:

```bash
pip install numpy nnfs
```

## How to run

Run any script with Python, for example:

```bash
python 1.neuron.py
```

You can also try:

```bash
python 3.using_numpy_in_nn.py
python 6.ReLU_and_Softmax.py
python 7.full_neural_network_no_loss.py
```

## Learning goals

This project is intended for learners who want to understand:

- how neurons compute outputs
- how weights and biases influence results
- how NumPy simplifies neural network math
- how layers and activation functions fit together
- how a basic network produces predictions

## Notes

These examples are intentionally simple and educational. They are designed to build intuition rather than serve as a production-ready neural network implementation.
