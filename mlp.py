#! /usr/bin/env python
# -*- encoding: utf-8 -*-

import sys
from argparse import ArgumentParser
from random import sample

import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST

class DenseLayer:
    def __init__(self, neurons, features,
                 activation_function=lambda n: 1/(1+np.exp(-n))):
        self.f = activation_function
        self.W = init_weights((neurons, features))
        self.b = init_weights((neurons,))
        self.Z = None
        self.A = None

    def linear(self, w, x):
        return w.T @ x + b

    def sigmoid(self, Z):
        return self.f(Z)

    def forward(self, n, x):
        self.Z = linear(self.W[:, n], x)
        return self.sigmoid(self.Z)

    def update(self, delta, grad, learning_rate):
        self.w = self.w - grad * learning_rate
        self.b = self.b - np.sum(delta, axis=0, keepdims=True) * learning_rate


class MultiLayerPerceptron:
    def __init__(self, train_x, train_y, test_x, test_y,
                 learning_rate=0.01, epochs=1, hidden_layers=None,
                 epsilon=1.0e-6):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.epochs = epochs
        self.features = len(train_x[0])
        self.output = len(set(test_y))
        self.params = self.init_params(hidden_layers)
        self.epsilon = epsilon

    def init_params(self, hidden_layers):
        params = {}
        if hidden_layers:
            params['W0'] = np.random.random((self.features, hidden_layers[0]))
            params['b0'] = np.random.random((1,  hidden_layers[0]))
            if len(hidden_layers) > 1:
                for i in range(1, len(hidden_layers)):
                    params[f'W{i}'] = np.random.random((hidden_layers[i-1], hidden_layers[i]))
                    params[f'b{i}'] = np.random.random((1, hidden_layers[i]))
        
            

    def fit(self):
        errors = []
        for i in range(self.epochs):
            
        
    def epoch(self):
        for n in range(output_dims):
            for x, d in zip(self.train_x, self.train_y):
                for layer in self.layers:
                    



def logistic(z):
    return 1/(1 + np.exp(-z))


def logistic_deriv(z):
    return sigmoid(z) * (1 - sigmoid(z))


def relu(z):
    return max(0, z)


def softplus(z):
    return np.log(1 + np.exp(z))


def softplus_deriv(z):
    return sigmoid(z)


def tanh(z):
    return (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))


def heaviside(z):
    if z > 0:
        return 1
    return 0


def get_data(ntrain=None, ntest=None, threshold=None, dataset='digits'):
    if dataset not in {"mnist", "balanced", "digits",
                       "bymerge", "letters", "byclass"}:
        print("Invalid dataset")
        return
    mndata = MNIST("./emnist_data/")
    mndata.select_emnist(dataset)
    train_x, train_y = mndata.load_training()
    test_x, test_y = mndata.load_testing()
    categories = len(set(test_y))
    if ntrain and 0 < ntrain < len(train_x) and ntest and 0 < ntest < len(test_x):
        train_idx = sample([n for n in range(ntrain)], ntrain)
        test_idx = sample([n for n in range(ntest)], ntest)
        train_x = [train_x[i] for i in train_idx]
        train_y = [train_y[i] for i in train_idx]
        test_x = [test_x[i] for i in test_idx]
        test_y = [test_y[i] for i in test_idx]

    if threshold and 0 <= threshold <= 255:
        train_x = np.array([binary(x, threshold) for x in train_x])
        test_x = np.array([binary(x, threshold) for x in test_x])
    else:
        train_x = np.array([normalize(x) for x in train_x])
        test_x = np.array([normalize(x) for x in test_x])

    if dataset == 'letters':
        train_y = np.subtract(train_y, 1)
        test_y = np.subtract(test_y, 1)

    train_y = [onehot(y, categories) for y in train_y]

    return train_x, train_y, test_x, test_y


def normalize(X):
    mn = min(X)
    mx = max(X)
    return np.array([(x - mn) / (mx - mn) for x in X])


def binary(x, threshold):
    b = np.zeros(x.shape)
    for i in range(len(b)):
        if x[i] > threshold:
            b[i] = 1
    return b


def onehot(n, total):
    vec = np.zeros(total)
    vec[n] = 1
    return vec


def init_weights(shape, zero=False):
    if zero:
        return np.zeros(shape)
    return np.random.random(shape)


def setup_argparser():
    parser = ArgumentParser()
    return parser


if __name__ == '__main__':
    sys.exit(0)
