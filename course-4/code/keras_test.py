"""Keras test code."""

from keras.models import Sequential
from keras.layers import Dense
import numpy as np


model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=1, activation=None))

model.compile(loss='mean_squared_error', optimizer='sgd')

# x_train and y_train are Numpy arrays
x_train = np.random.randn(1000, 100)
y_train = np.random.randn(1000, 1)
model.fit(x_train, y_train, epochs=5, batch_size=32)
