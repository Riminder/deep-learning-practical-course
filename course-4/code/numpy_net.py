"""Numpy code for a one layer neural network."""

import numpy as np


N = 3
D = 2

x = np.random.randn(N, D)
y = np.random.randn(N, 1)
w = np.random.randn(D, 1)

res = np.matmul(x, w)
loss = 0.5 * np.sum(np.square(res - y))

grad_w = x.T.dot(res - y)
