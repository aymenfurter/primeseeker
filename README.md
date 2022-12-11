Primeseeker - Prime Number Predictor
======================

This project contains a model that aims to predict whether or not a number is prime. The model is implemented using PyTorch, and it consists of a feedforward neural network with three hidden layers. So far the model usually ends up reporting that no number is prime (which is true the majority of the time). The training data generating may need some tweaking.

Getting Started
---------------

To use this model, you will need to have PyTorch installed. You can install PyTorch by following the instructions on the [official website](https://pytorch.org/).

Once you have PyTorch installed, you can run the `main.py` script to train and evaluate the model.

`python main.py`

Model Details
-------------

The model is a feedforward neural network with three hidden layers. The input to the model is a 5-dimensional vector representing various properties of a given number.
