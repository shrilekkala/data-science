import numpy as np
import tensorflow as tf


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

model = Sequential([
     Dense(400, activation='tanh', input_shape=(3072,)),
     Dense(400, activation='tanh'),
     Dense(400, activation='tanh'),
     Dense(400, activation='tanh'),
     Dense(400, activation='tanh'),
     Dense(10, activation='softmax')])

model.summary()

model2 = Sequential([
     Flatten(input_shape=(32, 32, 3)),
     Dense(12, activation='tanh'),
     Dense(10, activation='softmax')])

model2.summary()

model3 = Sequential([
     Flatten(input_shape=(32, 32, 3)),
     Dense(400, activation='tanh'),
     Dense(400, activation='tanh'),
     Dense(400, activation='tanh'),
     Dense(400, activation='tanh'),
     Dense(400, activation='tanh'),
     Dense(10, activation='softmax')])

model3.summary()
model3.count_params()
