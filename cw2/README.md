# Neural Network Image Classification (CIFAR-10)

This project implements and compares two approaches for image classification on **CIFAR-10**:

1) A **multi-layer perceptron (MLP) built from scratch** in NumPy (forward pass, backprop, softmax + cross-entropy, mini-batch training)  
2) A **convolutional neural network (CNN)** implemented in TensorFlow/Keras as a stronger baseline

The notebook explores optimization choices (learning rate, epochs), evaluates models on a held-out validation set, and compares accuracy, loss, runtime, and parameter count.

---

## Dataset
- **CIFAR-10** (50,000 training images, 10,000 test images)
- Images are **32×32 RGB** across **10 classes**
- Loaded via `tf.keras.datasets.cifar10`

---

## Summary
### 1) MLP from scratch (NumPy)
- Implements:
  - weight initialization
  - tanh activations
  - softmax output
  - cross-entropy loss
  - backpropagation gradients
  - mini-batch SGD training (batch size 128)
- Architecture (from scratch implementation):
  - 5 hidden layers, **400** neurons each (tanh)
  - 10-class softmax output

### 2) CNN (TensorFlow/Keras)
A compact CNN baseline:

- Conv(32) → MaxPool  
- Conv(64) → MaxPool  
- Conv(64) → Flatten  
- Dense(64) → Dense(10 softmax)

Includes experiments with regularization/dropout and early stopping.

---

## Results (from notebook runs)
- **MLP (scratch)** best validation accuracy: **~0.533**
- **CNN (Keras)** best validation accuracy: **~0.719**
- The CNN improves generalization substantially with a modest increase in training time.

(Exact results depend on random initialization and training settings.)

---

## How to Run
Open the notebook and run all cells:

**Notebook:** [`cw2/Data Science Coursework 2.ipynb`](https://github.com/shrilekkala/data-science/blob/main/cw2/Data%20Science%20Coursework%202.ipynb)

## Notes
This was completed as part of a Data Science coursework project. 
The focus is on understanding implementation details (MLP from scratch), experimentation, and model comparison rather than maximizing leaderboard accuracy.
