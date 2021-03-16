import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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

"""
Section 1.1
"""
# Convert the data to numpy
x_train_numpy = np.array(x_train)
x_val_numpy = np.array(x_val)
y_train_numpy = np.array(y_train)
y_val_numpy = np.array(y_val)

# Stack the x and y data



## Example Image
image = x_train[1024]
#image = tf.io.decode_jpeg(image, channels=3)
plt.figure(figsize=(5, 5))
plt.imshow(image)
plt.axis('off')
plt.show()

## load the data into tf.data.Dataset objects
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))

"""
Source:
https://datascience.stackexchange.com/questions/47623/how-feed-a-numpy-array-in-batches-in-keras
"""
def databatch(batch_size, x_data, y_data):
    n = x_data.shape[0]
    indices = np.arange(n)
    
    # shuffle before each epoch
    np.random.shuffle(indices)
    x_data = x_data[indices]
    y_data = y_data[indices]
    
    batches = [(x_data[k:k+batch_size], y_data[k:k+batch_size]) for k in range(0, n, batch_size)]
    
    # make the last batch the same size as the others by including elements from the start
    k = range(0, 50000, 128)[-1]
    last_batch_x = np.array([x_data[i%n] for i in range(k, k+batch_size)])
    last_batch_y = np.array([y_data[i%n] for i in range(k, k+batch_size)])
    
    batches[-1] = (last_batch_x, last_batch_y)
    
    return batches           

## Should return 391 batches
batches = databatch(128, x_train_numpy, y_train_numpy)


## Functions for use in SGD
def grad(x, y, beta0, beta):
    
    # x: K x D array of inputs
    # y: K x 1 array of outputs
    # beta0: Length 1 1-D array for bias parameter
    # beta: D x 1 array of parameters
    # returns: tuple of gradients for beta0 (length 1 1-D array) and beta (D x 1 array)
    
    g = (y - beta0 - x @ beta)  ## <-- K x 1 array
    return -g.mean(axis=0), -(x.T * g).mean(axis=1)


# Run the SGD algorithm

iterations = len(batches)
losses = []
learning_rate = 0.01

def mse_loss(x, y, beta0, beta):
    
    # x: K x D array of inputs
    # y: K x 1 array of outputs
    # beta0: Length 1 1-D array for bias parameter
    # beta: D x 1 array of parameters
    # returns: MSE computed on this batch of inputs and outputs; K x 1 array
    
    return 0.5 * ((y - beta0 - x @ beta)**2).mean()

beta0 = np.array([1.0])
beta = np.ones(3072) * -0.5

for iteration in range(iterations):
    # Obtain the batch
    x_batch, y_batch = batches[iteration]
    
    # print(x_batch.shape)
    
    # Flatten the input batches into a 2d array
    x_batch_flat = np.reshape(x_batch, (128, 3072), order='C')
    
    # Flatten the output batch by converting the vector of labels into a number
    y_batch_flat = np.argmax(y_batch, axis = 1)
    
    losses.append(mse_loss(x_batch_flat, y_batch_flat, beta0, beta))
    beta0_grad, beta_grad = grad(x_batch_flat, y_batch_flat, beta0, beta)
    beta0 -= learning_rate * beta0_grad
    beta -= learning_rate * beta_grad
    
    
def dense(h, W, b):
    
    # h: K x h_in array of inputs
    # W: h_in x h_out array for kernel matrix parameters
    # b: Length h_out 1-D array for bias parameters
    # returns: K x h_out output array 
    
    return b + h @ W ## <-- EDIT THIS LINE - DONE

def output_error(y_batch, a2):
    
    # y_batch: K x 1 array of data outputs
    # a2: K x 1 array of output pre-activations
    # returns: K x 1 array of output errors 
    
    return a2 - y_batch

def tanh(h):
    return np.tanh(h)

## EDIT THIS FUNCTION - DONE
def activation_derivative(a1):
    
    # a1: K x 64 array of hidden layer pre-activations
    # returns: K x 64 array of diagonal elements  
    
    ## Note the derivative of tanh(x) is 1 - tanh^2(x)
    
    return 1 - np.square(np.tanh(a1))
    
def grads(delta1, delta2, h0, h1):
    
    # delta1: K x 64 array of hidden layer errors
    # delta2: K x 1 array of output errors
    # h0: K x 6 array of inputs
    # h1: K x 64 array of hidden layer post-activations
    # returns: tuple of arrays of shape (6 x 64), (64,), (64 x 1), (1,) for gradients
    
    ## Obtain the gradients
    grad_W0 = delta1[:, np.newaxis, :] * h0[:, :, np.newaxis]
    grad_b0 = delta1
    grad_W1 = delta2[:, np.newaxis, :] * h1[:, :, np.newaxis]
    grad_b1 = delta2
    
    ## Average them over the batch size
    grad_W0 = tf.reduce_mean(grad_W0, axis=0)
    grad_b0 = tf.reduce_mean(grad_b0, axis=0)
    grad_W1 = tf.reduce_mean(grad_W1, axis=0)
    grad_b1 = tf.reduce_mean(grad_b1, axis=0)
    
    return grad_W0, grad_b0, grad_W1, grad_b1