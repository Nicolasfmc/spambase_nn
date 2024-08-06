# Spam Email Classification Using Neural Networks

This project applies neural networks to classify emails as spam or not spam using the UCI Spambase dataset. The project includes the implementation and evaluation of models with different activation functions.

## Table of Contents
- [Introduction](#introduction)
- [Model Implementation](#model-implementation)
- [Results](#results)

## Introduction

This project aims to implement and evaluate a neural network model to classify emails as spam or not spam using the UCI Spambase dataset (ID 94).

## Model Implementation

A multi-layer neural network model was implemented using the TensorFlow library in Python. The model was trained with ReLU and sigmoid activation functions. The primary difference between the models is the activation function used in the hidden layers.

For the ReLU model, activation was done as follows:
```python
train1 = tf.nn.relu(neuron(X_train, weights1, bias1))
train2 = tf.nn.relu(neuron(train1, weights2, bias2))
```

For the sigmoid model, activation was done as follows:
```python
train1 = tf.sigmoid(neuron(X_train, weights1, bias1))
train2 = tf.sigmoid(neuron(train1, weights2, bias2))
```

Other details of the code, including weight and bias initialization, loss function, optimizer, and training loop, remain the same for both implementations.

## Results

The results show that the ReLU activation function achieved a better accuracy rate compared to the sigmoid function, as observed in the loss and accuracy graphs per epoch.

### Results with Sigmoid Activation Function

The sigmoid activation function struggled to surpass the loss rate, as seen in the loss and accuracy graphs per epoch. Even increasing the number of training epochs, the accuracy rate never outperformed the loss rate.

#### Sigmoid - 1000 epochs
![Sigmoid - 1000 Epochs](assets/spambase_3sigmoid_1000.png)
#### Sigmoid - 2500 epochs
![Sigmoid - 2500 Epochs](assets/spambase_3sigmoid_2500.png)

### Results with ReLU Activation Function

In comparison, the ReLU activation function achieved a better accuracy rate, maintaining a high accuracy of 0.84 parallel to a cost of 0.41, with 1 as the maximum comparative result.

#### ReLU - 1000 epochs
![ReLU - 1000 Epochs](assets/spambase_3relu_1000.png)
#### ReLU - 2500 epochs
![ReLU - 2500 Epochs](assets/spambase_3relu_2500.png)
