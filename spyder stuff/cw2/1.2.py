import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

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

## One-Hot labels to integer labels
y_train_labels = tf.argmax(y_train, axis=1)
y_val_labels = tf.argmax(y_val, axis=1)

## load the data into tf.data.Dataset objects
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))

## batch the datasets
batch_size = 128
train_dataset_batched = train_dataset.batch(batch_size)
val_dataset_batched = val_dataset.batch(batch_size)
train_dataset_batched.element_spec

# Load the list of label names
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Plot a randomly selected example from each class
n_rows, n_cols = 2, 5
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8))
fig.subplots_adjust(hspace=0.2, wspace=0.1)


for l, label_name in enumerate(classes):
    row = l // n_cols
    col = l % n_cols
    
    print(row, col)
    
    inx = np.where(y_train_labels == l)[0]
    i = np.random.choice(inx)
    x_example = x_train[i]
    
    axes[row, col].imshow(x_example)
    axes[row, col].get_xaxis().set_visible(False)
    axes[row, col].get_yaxis().set_visible(False)
    axes[row, col].set_title(label_name)

plt.show()

"""
Section 1.2.1
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

model_1 = Sequential([
     Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
     MaxPool2D((2, 2)),
     Conv2D(32, (3, 3), activation='relu'),
     MaxPool2D((2, 2)),
     Conv2D(64, (3, 3), activation='relu'),
     MaxPool2D((2, 2)),
     Flatten(),
     Dense(64, activation='relu'),
     Dense(10, activation='softmax')])

model_1.summary()

def train_CNN(model, callback=None):
    start = time.time()
    
    # Note Categorical Cross Entropy instead of sparse as we use one-hot encodings
    sgd = tf.keras.optimizers.SGD(learning_rate=0.1)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    
    model.compile(optimizer=sgd, loss=loss_fn, metrics=['categorical_accuracy'])
    
    # Train the model using the training set
    if callback:
        history = model.fit(train_dataset_batched, epochs=40, validation_data=val_dataset_batched, verbose=0,
                            callbacks = [callback])
    else:
        history = model.fit(train_dataset_batched, epochs=40, validation_data=val_dataset_batched, verbose=2)
    
    # Calculate the time taken
    time_diff = time.time() - start
    train_time = time.strftime("%H:%M:%S", time.gmtime(time_diff))
    
    return history, train_time

history_1, train_time_1 = train_CNN(model_1)
print('Total Training Time  : ', train_time_1)

def CNN_plots(history, reg_info):
    # PLot of losses and accuracies
    reg_info = "None"
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
    
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    
    x_axis = np.arange(40) + 1
    ax0.set_title("Loss vs Number of Epochs," + 
                  "[CNN, $\eta$ = 0.1, Regularizer = " + reg_info + "]")
    ax0.plot(x_axis, train_loss, label = "Training Set")
    ax0.plot(x_axis, val_loss, label = "Validation Set")
    ax0.legend()
    ax0.grid()
    ax0.set_xlabel("Epoch Number")
    ax0.set_ylabel("Categorical Cross-Entropy Loss")
    
    ax1.set_title("Accuracy vs Number of Epochs ," + 
                  "[CNN, $\eta$ = 0.1, Regularizer = " + reg_info + "]")
    ax1.plot(x_axis, train_acc, label = "Training Set")
    ax1.plot(x_axis, val_acc, label = "Validation Set")
    ax1.legend()
    ax1.grid()
    ax1.set_xlabel("Epoch Number")
    ax1.set_ylabel("Categorical Accuracy")
    plt.show()
    
    return

CNN_plots(history_1, "None")




# Also plot predicted categorical distributions? (histogram like - see TF tutorial)


"""
Section 1.2.2
"""
from tensorflow.keras import regularizers

model_2 = Sequential([
     Conv2D(32, (3, 3), kernel_regularizer=regularizers.l2(5e-3), activation='relu', input_shape=(32, 32, 3)),
     MaxPool2D((2, 2)),
     Conv2D(32, (3, 3), kernel_regularizer=regularizers.l2(5e-3), activation='relu'),
     MaxPool2D((2, 2)),
     Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(5e-3), activation='relu'),
     MaxPool2D((2, 2)),
     Flatten(),
     Dense(64, activation='relu'),
     Dense(10, activation='softmax')])

model_2.summary()

history_2, train_time_2 = train_CNN(model_2)
print('Total Training Time  : ', train_time_2)

CNN_plots(history_2, "L2 Regularisation, coef: $5 /cdot 10^{-3}$")


"""
Section 1.2.3
"""
"""a"""
from tensorflow.keras.layers import Dropout

# Dropout in between fully connected layers (justify), instead of Conv layers

model_3= Sequential([
     Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
     MaxPool2D((2, 2)),
     Conv2D(32, (3, 3), activation='relu'),
     MaxPool2D((2, 2)),
     Conv2D(64, (3, 3), activation='relu'),
     MaxPool2D((2, 2)),
     Flatten(),
     Dense(64, activation='relu'),
     Dropout(0.5),
     Dense(10, activation='softmax')])

model_3.summary()

history_3, train_time_3 = train_CNN(model_3)
print('Total Training Time  : ', train_time_3)

CNN_plots(history_3, "Dropout (rate 0.5)")


"""b"""
earlystopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=10)

model_4 = Sequential([
     Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
     MaxPool2D((2, 2)),
     Conv2D(32, (3, 3), activation='relu'),
     MaxPool2D((2, 2)),
     Conv2D(64, (3, 3), activation='relu'),
     MaxPool2D((2, 2)),
     Flatten(),
     Dense(64, activation='relu'),
     Dense(10, activation='softmax')])

model_4.summary()

history_4, train_time_4 = train_CNN(model_4, callback = earlystopping)

print('Total Training Time  : ', train_time_4)

CNN_plots(history_4, "Early Stopping")

"""
Model Predictions
"""

# Get predictions from the model
for images, labels in val_dataset_batched.take(1):
    preds = model_1.predict(images)


total_num_images = preds.shape[0]

random_inx = np.random.choice(total_num_images, 4)
random_preds = preds[random_inx, ...]
random_test_images = images.numpy()[random_inx, ...]
random_test_labels = labels.numpy()[random_inx, ...]

fig, axes = plt.subplots(4, 2, figsize=(16, 12))
fig.subplots_adjust(hspace=0.4, wspace=-0.2)

for i, (prediction, image, label) in enumerate(zip(random_preds, random_test_images, random_test_labels)):
    axes[i, 0].imshow(np.squeeze(image))
    axes[i, 0].get_xaxis().set_visible(False)
    axes[i, 0].get_yaxis().set_visible(False)
    axes[i, 0].text(10., -1.5, f'{classes[label]}')
    axes[i, 1].bar(np.arange(len(prediction)), prediction)
    axes[i, 1].set_xticks(np.arange(len(prediction)))
    axes[i, 1].set_title(f"Categorical distribution. Model prediction: {classes[np.argmax(prediction)]}")
plt.show()



