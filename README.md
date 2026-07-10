# Neural Network from Scratch

This folder contains a small, beginner-friendly collection of Python scripts that demonstrate how neural networks work without using a high-level framework. The examples focus on the core building blocks: basic neurons, layers, matrix operations, and activation functions.

## What this project covers

These scripts introduce the fundamentals of neural networks, including:

- A single neuron and its computation
- Multiple neurons in parallel
- Vectorized operations with NumPy
- A simple layer of neurons
- A class-based dense layer implementation
- ReLU activation

## Project structure

- 1.neuron.py - Demonstrates a single neuron with manual computation.
- 2.multiple_neurons.py - Shows how multiple neurons are computed together.
- 3.using_numpy_in_nn.py - Uses NumPy for efficient matrix-based neural network calculations.
- 4.layer_of_nurons.py - Builds a simple two-layer neural network example.
- 5.using_classes_for_nn.py - Implements a reusable dense layer using a Python class.
- 6.ReLU_activation.py - Demonstrates the ReLU activation function.
- basics/ - Extra NumPy examples for broadcasting and summation.
- data/ - Sample datasets used for learning and experimentation.

## Requirements

Make sure you have Python installed, then install NumPy if needed:

```bash
pip install numpy
```

If you want to run the class-based example, you may also need the nnfs package:

```bash
pip install nnfs
```

## How to run

Run any script with Python, for example:

```bash
python 1.neuron.py
```

You can also try:

```bash
python 3.using_numpy_in_nn.py
python 4.layer_of_nurons.py
```

## Learning goals

This project is intended for learners who want to understand:

- how neurons compute outputs
- how weights and biases affect results
- how NumPy simplifies neural network math
- how layers and activation functions fit together

## Notes

These examples are intentionally simple and educational. They are meant to build intuition rather than serve as a production-ready neural network implementation.

