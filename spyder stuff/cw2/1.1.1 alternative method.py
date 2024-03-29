import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

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
# Convert the data into 2D numpy arrays
X_train_np = np.reshape(np.array(x_train), (50000, 3072), order='C').T
X_test_np = np.reshape(np.array(x_val), (10000, 3072), order='C').T
y_train_np = np.array(y_train).T
y_test_np = np.array(y_val).T

# Convert the vector of labels of 0s and 1s into an integer between 0 and 9
y_train_np_labels = np.argmax(y_train_np, axis = 0)
y_test_np_labels = np.argmax(y_test_np, axis = 0)


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

## Create the activation functins for the forward pass

def Tanh(h):
    return np.tanh(h)

def SoftMax(X):
    prob = np.exp(X) /np.sum(np.exp(X), axis=0)
    return prob

## Create the functions of the derivatives of the activation functions (for backward pass)
def dTanh(a1):
    # Note the derivative of tanh(x) is 1 - tanh^2(x)
    return 1 - np.square(np.tanh(a1))


## Define the loss function (no regulatisation)
def CrossEntropyLoss(y, y_hat):
    M = y.shape[1]
    Cost = - np.sum(y * np.log(y_hat)) / M
    return Cost

## Define the layer transformation functions (pre -> post activation)
def dense(h, W, b):
    return b + W @ h

def output_error(y_batch, a2):
    return a2 - y_batch

## Forward and Backward Pass functions
def forward_pass(X, parameters):
    # create a dictionary to store the pre and post activations
    forwardPass = {}
    
    # Five Hidden Layers
    forwardPass['a1'] = dense(X, parameters['W0'], parameters['b0'])
    forwardPass['h1'] = Tanh(forwardPass['a1'])
    
    forwardPass['a2'] = dense(forwardPass['h1'], parameters['W1'], parameters['b1'])
    forwardPass['h2'] = Tanh(forwardPass['a2'])
    
    forwardPass['a3'] = dense(forwardPass['h2'], parameters['W2'], parameters['b2'])
    forwardPass['h3'] = Tanh(forwardPass['a3'])
    
    forwardPass['a4'] = dense(forwardPass['h3'], parameters['W3'], parameters['b3'])
    forwardPass['h4'] = Tanh(forwardPass['a4'])
    
    forwardPass['a5'] = dense(forwardPass['h4'], parameters['W4'], parameters['b4'])
    forwardPass['h5'] = Tanh(forwardPass['a5'])
    
    # Output Layer
    forwardPass['a6'] = dense(forwardPass['h5'], parameters['W5'], parameters['b5'])
    forwardPass['h6'] = SoftMax(forwardPass['a6'])
    
    return forwardPass
    
def back_propagate(X, y, forwardPass, parameters):
    M = X.shape[1]
    
    # create a dictionary to store the gradients
    gradient = {}
    
    gradient['delta6'] = output_error(y, forwardPass['h6'])
    gradient['d_W5'] = gradient['delta6'] @ forwardPass['h5'].T / M
    gradient['d_b5'] = np.sum(gradient['delta6'], axis=1, keepdims=True) / M
    gradient['d_h5'] = parameters['W5'].T @ gradient['delta6']
    
    gradient['delta5'] = gradient['d_h5'] * dTanh(forwardPass['a5'])
    gradient['d_W4'] = gradient['delta5'] @ forwardPass['h4'].T / M
    gradient['d_b4'] = np.sum(gradient['delta5'], axis=1, keepdims=True) / M
    gradient['d_h4'] = parameters['W4'].T @ gradient['delta5']
    
    gradient['delta4'] = gradient['d_h4'] * dTanh(forwardPass['a4'])
    gradient['d_W3'] = gradient['delta4'] @ forwardPass['h3'].T / M
    gradient['d_b3'] = np.sum(gradient['delta4'], axis=1, keepdims=True) / M
    gradient['d_h3'] = parameters['W3'].T @ gradient['delta4']
    
    gradient['delta3'] = gradient['d_h3'] * dTanh(forwardPass['a3'])
    gradient['d_W2'] = gradient['delta3'] @ forwardPass['h2'].T / M
    gradient['d_b2'] = np.sum(gradient['delta3'], axis=1, keepdims=True) / M
    gradient['d_h2'] = parameters['W2'].T @ gradient['delta3']
    
    gradient['delta2'] = gradient['d_h2'] * dTanh(forwardPass['a2'])
    gradient['d_W1'] = gradient['delta2'] @ forwardPass['h1'].T / M
    gradient['d_b1'] = np.sum(gradient['delta2'], axis=1, keepdims=True) / M
    gradient['d_h1'] = parameters['W1'].T @ gradient['delta2']
    
    gradient['delta1'] = gradient['d_h1'] * dTanh(forwardPass['a1'])
    gradient['d_W0'] = gradient['delta1'] @ X.T / M
    gradient['d_b0'] = np.sum(gradient['delta1']) / M

    return gradient

def SGD_updater(parameters, gradient, learning_rate):
    # Create a dictionary to store the updated parameters
    new_parameters = {}
    
    new_parameters['W5'] = parameters['W5'] - learning_rate * gradient['d_W5']
    new_parameters['b5'] = parameters['b5'] - learning_rate * gradient['d_b5']
    
    new_parameters['W4'] = parameters['W4'] - learning_rate * gradient['d_W4']
    new_parameters['b4'] = parameters['b4'] - learning_rate * gradient['d_b4']

    new_parameters['W3'] = parameters['W3'] - learning_rate * gradient['d_W3']
    new_parameters['b3'] = parameters['b3'] - learning_rate * gradient['d_b3']
    
    new_parameters['W2'] = parameters['W2'] - learning_rate * gradient['d_W2']
    new_parameters['b2'] = parameters['b2'] - learning_rate * gradient['d_b2']
    
    new_parameters['W1'] = parameters['W1'] - learning_rate * gradient['d_W1']
    new_parameters['b1'] = parameters['b1'] - learning_rate * gradient['d_b1']
    
    new_parameters['W0'] = parameters['W0'] - learning_rate * gradient['d_W0']
    new_parameters['b0'] = parameters['b0'] - learning_rate * gradient['d_b0']
    
    return new_parameters

## Create a function to classify a set of inputs using the model

def classify(X, parameters):
    a1 = dense(X, parameters['W0'], parameters['b0'])
    h1 = Tanh(a1)
    
    a2 = dense(h1, parameters['W1'], parameters['b1'])
    h2 = Tanh(a2)
    
    a3 = dense(h2, parameters['W2'], parameters['b2'])
    h3 = Tanh(a3)
    
    a4 = dense(h3, parameters['W3'], parameters['b3'])
    h4 = Tanh(a4)
    
    a5 = dense(h4, parameters['W4'], parameters['b4'])
    h5 = Tanh(a5)
    
    a6 = dense(h5, parameters['W5'], parameters['b5'])
    h6 = SoftMax(a6)
    
    preds = np.argmax(h6, axis=0)
    return h6, preds


# Random Seed
np.random.seed(7)


# Initialise the set of parameters
def initial_parameters(num_h, D):
    parameters = {'W0': np.random.randn(num_h, D) * 0.02,
                  'b0': np.zeros((num_h, 1)),
                  
                  'W1': np.random.randn(num_h, num_h) * 0.05,
                  'b1': np.zeros((num_h, 1)),
                    
                  'W2': np.random.randn(num_h, num_h) * 0.05,
                  'b2': np.zeros((num_h, 1)),
                     
                  'W3': np.random.randn(num_h, num_h) * 0.05,
                  'b3': np.zeros((num_h, 1)),
                     
                  'W4': np.random.randn(num_h, num_h) * 0.05,
                  'b4': np.zeros((num_h, 1)),
                     
                  'W5': np.random.randn(10, num_h) * 0.05,
                  'b5': np.zeros((10, 1))}
    
    return parameters



## Create a function to train the MLP with epochs and learning rates as hyperparameters
def MLP(num_epochs, l_rate):
    
    # Create lists to store accuracies and losses over the epochs
    Accuracies = []
    Losses = []
    
    # Obtain the start time  
    start = time.time()
    
    # Number of descriptors
    D = X_train_np.shape[0]
    
    # Set the number of neurons per hidden layer
    num_h = 400
    
    # Get the initialised set of parameters
    Params = initial_parameters(num_h, D)
    
    
    ## Loop over the epochs
    for i in range(num_epochs):
        
        # Obtain new randomly sampled batches
        ## Should return 391 batches (for 50,000 data points)
        batches = databatch(128, X_train_np.T, y_train_np.T)
        
        # Loop over each batch
        for x_batch, y_batch in batches:
            X = x_batch.T
            y = y_batch.T
        
            # Forward Pass
            forwardPass = forward_pass(X, Params)
            
            # Calculate the loss
            # ce_loss = CrossEntropyLoss(y, forwardPass['h6'])
            
            # Back Propagate to find the gradients
            gradient = back_propagate(X, y, forwardPass, Params)
        
            # Update the Weights according to SGD
            Params = SGD_updater(Params, gradient, l_rate)
        
        # Obtain the predictions using the model for the training and validation sets
        y_train_softmax, y_train_hat = classify(X_train_np, Params)
        y_val_softmax, y_val_hat = classify(X_test_np, Params)
        
        # Compute the accuracies
        train_acc = np.sum(y_train_hat == y_train_np_labels) / len(y_train_np_labels)
        val_acc = np.sum(y_val_hat == y_test_np_labels) / len(y_test_np_labels)
        
        # Compute the losses
        train_loss = CrossEntropyLoss(y_train_np, y_train_softmax)
        val_loss = CrossEntropyLoss(y_test_np, y_val_softmax)
        
        # Store the the metrics in the lists
        Losses.append((train_loss, val_loss))
        Accuracies.append(((train_acc, val_acc)))
        
        # Print the progress
        if i%5 == 4:
            print("Epoch " + str(i+1) + "/ 40")
    
    # Calculate the time taken
    time_diff = time.time() - start
    train_time = time.strftime("%H:%M:%S", time.gmtime(time_diff))
    
    # Return the metrics
    return Losses, Accuracies, train_time

# Store the metrics in a dictionary
MLP_Metrics = {}
MLP_Metrics["40, 0.01"] = MLP(40, 0.01)

print("Final Validation Loss           :", MLP_Metrics["40, 0.01"][0][-1][1])
print('Final Validation Accuracy       :', MLP_Metrics["40, 0.01"][1][-1][1])
print('Total Training Time  : ', MLP_Metrics["40, 0.01"][2])


# Create a function to plot the required figures
def MLP_plots(key, lrate):
    # Obtain the metrics
    train_loss = np.array(MLP_Metrics[key][0])[:, 0]
    val_loss = np.array(MLP_Metrics[key][0])[:, 1]
    train_acc = np.array(MLP_Metrics[key][1])[:, 0]
    val_acc = np.array(MLP_Metrics[key][1])[:, 1]
    
    # Plot the metrics
    x_axis = np.arange(40) + 1
    plt.title("Plot of Cross-Entropy losses over Number of Epochs, [Learning Rate = " + lrate + "]")
    plt.plot(x_axis, train_loss, label = "Training Set")
    plt.plot(x_axis, val_loss, label = "Validation Set")
    plt.legend()
    plt.grid()
    plt.xlabel("Epoch Number")
    plt.ylabel("Loss")
    plt.show()
    
    plt.title("Plot of Accuracies over Number of Epochs [Learning Rate = " + lrate + "]")
    plt.plot(x_axis, train_acc, label = "Training Set")
    plt.plot(x_axis, val_acc, label = "Validation Set")
    plt.legend()
    plt.grid()
    plt.xlabel("Epoch Number")
    plt.ylabel("Accuracy")
    plt.show()
    
    return

MLP_plots("40, 0.01", "0.01")

"""
# Train the MLP for 40 epochs and different learning rates
MLP_Metrics["40, 0.0001"] = MLP(40, 0.0001)
MLP_Metrics["40, 0.1"] = MLP(40, 0.1)

# Plot the metrics
MLP_plots("40, 0.0001", "0.0001")
MLP_plots("40, 0.1", "0.1")

# Train and plot the metrics for the MLP for 80 epochs with learning rate 0.01
MLP_Metrics["80, 0.01"] = MLP(80, 0.01)
MLP_plots("80, 0.01", "0.01")
"""

