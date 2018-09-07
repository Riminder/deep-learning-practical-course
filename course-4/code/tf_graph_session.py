"""Example code to use a graph and a session."""

import numpy as np
import tensorflow as tf

N, D = 3, 2

x = np.random.randn(N, D)
y = np.random.randn(N, 1)
w = np.random.randn(D, 1)

res = tf.matmul(x, w)
loss = 0.5 * tf.reduce_sum(tf.square(res - y))

sess = tf.Session()
print(sess.run(loss))
