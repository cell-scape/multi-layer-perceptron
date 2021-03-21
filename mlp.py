
#! /usr/bin/env python
# -*- encoding: utf-8 -*-

import sys
from argparse import ArgumentParser
from random import sample

import emnist
import numpy as np
import matplotlib.pyplot as plt


class MultiLayerPerceptron:
    def __init__(self, train_x, train_y, test_x, test_y,
                 learning_rate=0.01, epochs=1, hidden_layers=None,
                 epsilon=1.0e-6, activation_function=logistic):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.epochs = epochs
        self.features = len(train_x[0])
        self.output = len(set(test_y))
        if type(hidden_layers) == int:
            hidden_layers = [hidden_layers]
        self.hidden_layers = hidden_layers
        self.params = self.init_params(hidden_layers)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.sigmoid = activation_function
        self.errors = []
        self.accuracy = []
        self.mse = []
        self.letters = {n: {'correct': 0, 'incorrect': 0}
                        for n in range(self.output)}

    def init_params(self, hidden_layers):
        params = {}
        if hidden_layers:
            params['W0'] = np.random.random((self.features, hidden_layers[0]))
            params['b0'] = np.random.random((1,  hidden_layers[0]))
            if len(hidden_layers) > 1:
                i = 1
                while i < len(hidden_layers):
                    params[f'W{i}'] = np.random.random((params[f'W{i-1}'].shape[1], hidden_layers[i]))
                    params[f'b{i}'] = np.random.random((1, hidden_layers[i]))
                    i += 1
                params[f'W{i}'] = np.random.random((params[f'W{i-1}'].shape[1], self.output))
                params[f'b{i}'] = np.random.random((1, self.output))
            else:
                params['W1'] = np.random.random((params['W0'].shape[1], self.output))
                params['b1'] = np.random.random((1, self.output))
        else:
            params['W0'] = np.random.random((self.features, self.output))
            params['b0'] = np.random.random((1, self.output))
        return params

    def feedforward(self, X):
        z = linear(X, self.params['W0'], self.params['b0'])
        a = self.sigmoid(z)
        A = [a]
        for i in range(1, len(self.params) // 2):
            z = linear(a, self.params[f'W{i}'], self.params[f'b{i}'])
            a = self.sigmoid(z)
            A.append(a)
        return A

    def backprop(self, A):
        
        delta = (self.train_y - A[-1]) * A[-1] * (1 - A[-1])
        for i in reversed(range(1, len(self.params) // 2)):
            grad = A[i-1].T @ delta
            self.params[f'W{i}'] -= grad * self.learning_rate
            self.params[f'W{i}'] -= np.sum(delta, axis=0, keepdims=True) * self.learning_rate
            delta = (delta @ self.params[f'W{i}'].T) * A[i-1] * (1 - A[i-1])
        grad = self.train_x.T @ delta
        self.params['W0'] -= grad * self.learning_rate
        self.params['b0'] -= np.sum(delta, axis=0, keepdims=True) * self.learning_rate

    def epoch(self, X):
        A = self.feedforward(X)
        self.errors.append(A[-1])
        self.backprop(A)

    def train(self):
        for i in range(self.epochs):
            self.epoch(self.train_x)
            mse = cost(self.errors)
            self.mse.append(mse)
            if mse < self.epsilon:
                print(f"Reached epsilon {self.epsilon} at {i} epochs")
                break

    def test(self):
        A = self.feedforward(self.test_x)[-1]
        correct = 0
        for i, y in enumerate(self.test_y):
            yhat = np.argmax(A[i])
            if yhat == y:
                correct += 0
                self.letters[y]['correct'] += 1
            else:
                self.letters[y]['incorrect'] += 1
        return correct / len(self.test_y)

    def test_iterations(self):
        for _ in range(self.epochs):
            self.epoch(self.test_x)
            self.accuracy.append(self.test())

    def plot_mse(self):
        if self.mse == []:
            self.train()
        plt.plot(self.mse, label="mean squared error")
        plt.title("mean squared error vs. iterations")
        plt.legend()
        plt.show()

    def plot_accuracy(self):
        if self.accuracy == []:
            self.test_iterations()
        plt.plot(self.accuracy, label="accuracy")
        plt.title("accuracy vs. iterations")
        plt.legend()
        plt.show()

    def plot_letters(self):
        self.reset()
        self.train()
        _ = self.test()
        correct = [self.letters[n]['correct'] for n in range(self.output)]
        incorrect = [self.letters[n]['incorrect'] for n in range(self.output)]
        plt.bar(np.arange(len(correct)), correct, label='correct')
        plt.bar(np.arange(len(incorrect)), incorrect,
                bottom=correct, label='incorrect')
        plt.title("accuracy per character")
        plt.legend()
        plt.show()

    def reset(self):
        self.errors = []
        self.mse = []
        self.accuracy = []
        self.params = self.init_params(self.hidden_layers)
        self.letters = {n: {'correct': 0, 'incorrect': 0}
                        for n in range(self.output)}


def linear(X, W, b):
    return X @ W + b


def cost(a):
    return np.mean(np.power(a, 2)) / 2


def logistic(z):
    return 1/(1 + np.exp(-z))


def relu(z):
    return max(0, z)


def softplus(z):
    return np.log(1 + np.exp(z))


def softplus_deriv(z):
    return logistic(z)


def tanh(z):
    return (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))


def heaviside(z):
    if z > 0:
        return 1
    return 0


def softmax(z):
    return


def logloss(A):
    return


def get_data(ntrain=None, ntest=None, threshold=None, dataset='digits'):
    if dataset not in {"mnist", "balanced", "digits",
                       "bymerge", "letters", "byclass"}:
        print("Invalid dataset")
        return
    train_x, train_y = emnist.extract_training_samples(dataset)
    test_x, test_y = emnist.extract_test_samples(dataset)
    categories = len(set(test_y))
    if ntrain and 0 < ntrain < len(train_x) and ntest and 0 < ntest < len(test_x):
        train_idx = sample([n for n in range(ntrain)], ntrain)
        test_idx = sample([n for n in range(ntest)], ntest)
        train_x = [train_x[i] for i in train_idx]
        train_y = [train_y[i] for i in train_idx]
        test_x = [test_x[i] for i in test_idx]
        test_y = [test_y[i] for i in test_idx]

    train_x = [x.flatten() for x in train_x]
    test_x = [x.flatten() for x in test_x]

    if threshold and 0 <= threshold <= 255:
        train_x = np.array([binary(x, threshold) for x in train_x])
        test_x = np.array([binary(x, threshold) for x in test_x])
    else:
        train_x = np.array([normalize(x) for x in train_x])
        test_x = np.array([normalize(x) for x in test_x])

    if dataset == 'letters':
        train_y = np.subtract(train_y, 1)
        test_y = np.subtract(test_y, 1)

    train_y = np.array([onehot(y, categories) for y in train_y])
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


def setup_argparser():
    parser = ArgumentParser()
    return parser

                        
