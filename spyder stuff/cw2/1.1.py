#import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt

"""""""""""""""
Question 1.1.1
"""""""""""""""

def load_data ():
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_val = x_val.astype('float32') / 255
    
    # convert labels to categorical samples
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=10)
    return ((x_train, y_train), (x_val, y_val))

(x_train, y_train), (x_val, y_val) = load_data()

print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)
