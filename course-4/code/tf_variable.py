"""Example code to use a placeholder."""

import numpy as np
import tensorflow as tf

N, D = 3, 2

x = tf.placeholder(tf.float32, [N, D])
y = tf.placeholder(tf.float32, [N, 1])
w = np.random.randn(D, 1).astype(np.float32)
w = tf.Variable(w)

res = tf.matmul(x, w)
loss = 0.5 * tf.reduce_sum(tf.square(res - y))

sess = tf.Session()
sess.run(w.initializer)
feed_dict = {x: np.random.randn(N, D),
             y: np.random.randn(N, 1)}
print(sess.run(loss, feed_dict=feed_dict))
