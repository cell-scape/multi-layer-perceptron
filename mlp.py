#! /usr/bin/env python
# -*- encoding: utf-8 -*-

import sys
import argparse
from math import exp
from random import sample, random

import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST

class DenseLayer:
    def __init__(self, nneurons, nfeatures, activation_function=lambda n: 1/(1+exp(-n))):
        self.f = activation_function
        self.weights = init_weights((nneurons, nfeatures))
        self.bias = init_weights((nneurons, 1))

    def _apply_weights(self, w, x):
        return np.dot(np.transpose(w), x) + self.bias

    def _activation(self, z):
        return self.f(z)

    def _error(self, w, x, d):
        z = self._apply_weights(w, x)
        yhat = self._activation(z)
        return d - yhat

    def _update(self, w, x, lr, error):
        return w + lr * error * x


class MultiLayerPerceptron:
    def __init__(self, layers, ):
        pass


def sigmoid(z):
    return 1/(1 + exp(-z))


def relu(z):
    return max(0, z)


def tanh(z):
    return (exp(z) - exp(-z))/(exp(z) + exp(-z))


def heaviside(z):
    if z > 0:
        return 1
    return 0


def linear(z):
    return z


def get_data(ntrain=None, ntest=None, threshold=None, dataset='digits'):
    if dataset not in {"mnist", "balanced", "digits",
                       "bymerge", "letters", "byclass"}:
        print("Invalid dataset")
        return

    if dataset == 'digits':
        mndata = MNIST("./data")
    else:
        mndata = MNIST("./emnist_data/")
        mndata.select_emnist(dataset)

    train_x, train_y = mndata.load_training()
    test_x, test_y = mndata.load_testing()

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

    train_y = [onehot(y) for y in train_y]

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
    parser = argparse.ArgumentParser()
    return parser


if __name__ == '__main__':
    sys.exit(0)
