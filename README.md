# 🧠 Neural Network from Scratch (NumPy Only)

## 🚀 Overview

This project implements a **fully functional Neural Network from scratch using only NumPy**, without relying on any machine learning frameworks like TensorFlow or PyTorch.

The goal of this project is to deeply understand how neural networks work internally — including forward propagation, backpropagation, and weight updates.

---

## 🎯 Features

* Built using **pure NumPy**
* No high-level ML libraries used
* Implements:

  * Forward Propagation
  * Backpropagation
  * Gradient Descent
* Customizable architecture (layers, neurons)
* Easy-to-understand modular code

---

## 🧠 Concepts Covered

* Neurons, weights, and biases
* Activation functions (Sigmoid / ReLU / etc.)
* Loss functions
* Gradient descent optimization
* Chain rule & backpropagation

---

## 📂 Project Structure

```
neural-network-no-framework/
│
├── README.md
├── model.py          # Neural network implementation
├── train.py          # Training script
├── utils.py          # Helper functions (activations, loss, etc.)
└── data/             # (Optional) Dataset
```

---

## ⚙️ How It Works

### 1. Forward Propagation

Input data is passed through layers:

```
Z = W·X + b
A = activation(Z)
```

### 2. Loss Calculation

The model compares predicted output with actual output.

### 3. Backpropagation

Gradients are calculated using the chain rule to update weights.

### 4. Weight Update

```
W = W - learning_rate * dW
```

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/neural-network-no-framework.git
cd neural-network-no-framework
```

### 2. Install dependencies

```bash
pip install numpy
```

### 3. Run the model

```bash
python train.py
```

---

## 📊 Example Use Case

This neural network can be used for:

* Binary classification (e.g., Hired / Not Hired)
* Simple regression tasks
* Educational demonstrations of deep learning

---

## 🔥 Why This Project?

Most libraries abstract away the core logic of neural networks.
This project helps you:

* Understand **what actually happens under the hood**
* Build strong ML fundamentals
* Prepare for advanced AI/ML concepts

---

## 🛠️ Future Improvements

* Add multiple hidden layers (Deep NN)
* Implement different optimizers (Adam, RMSprop)
* Add visualization of loss curves
* Support batch training

---

## 🤝 Contributing

Feel free to fork this repository and improve it. Contributions are always welcome!

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 💡 Author

Built with 💻 and curiosity while learning AI/ML from scratch.
