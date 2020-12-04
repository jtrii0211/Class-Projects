from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
import json
from matplotlib import pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import sys, random



# Load in data
file = open(r'/home/john/ml/project/shipsnet.json')
dataset = json.load(file)
file.close()

dataset.keys()

#Filtering out images and labels from JSON object
x = np.array(dataset['data']).astype('uint8')
y = np.array(dataset['labels']).astype('uint8')

x.shape

#Visualizing Images with labels
plt.figure(figsize=(20,10))
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x[i].reshape(80, 80, 3), cmap=plt.cm.binary)
    plt.xlabel(classes[y[i]])

#The input shape of the images needs to be reshaped before being fed to the input layer of the CNN
X = x.reshape([-1, 3, 80, 80])

'''
This puts the labels into more categories in the event that you want to try the typical CNN 
architecture as shown below in a commented section of code
y = to_categorical(y, num_classes=2)
'''
# shuffle all indexes
indexes = np.arange(4000)
np.random.shuffle(indexes)

#Shuffling Images and Labels by same shuffled index
X = X[indexes].transpose([0,2,3,1])
Y = y[indexes]

#Normalization
X = X/255.0


#Splitting data into training and testing set
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2)


#Creating the Residual Units for the ResNet-34

class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            keras.layers.Conv2D(filters, 3, strides=strides,
                                padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters, 3, strides=1,
                                padding="same", use_bias=False),
            keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                keras.layers.Conv2D(filters, 1, strides=strides,
                                    padding="same", use_bias=False),
                keras.layers.BatchNormalization()]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)

# Creating the ResNet-34
cnn_model = tf.keras.Sequential()
cnn_model.add(keras.layers.Conv2D(64, 7, strides=2, input_shape=[80, 80, 3],
                             padding="same", use_bias=False))
cnn_model.add(keras.layers.BatchNormalization())
cnn_model.add(keras.layers.Activation("relu"))
cnn_model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"))
prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    cnn_model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters
cnn_model.add(keras.layers.GlobalAvgPool2D())
cnn_model.add(keras.layers.Flatten())
cnn_model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

# This was the code pulled from Kaggle that I used a baseline of comparison against the ResNet-34
# It is a typical CNN architecture for image classification
'''
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(80, 80, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #40x40
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #20x20
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #10x10
model.add(Dropout(0.25))

model.add(Conv2D(32, (10, 10), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #5x5
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))
'''
  

#Creating Checkpoint for saving weights
checkpoint_path = "resnet_cnn/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


# optimization setup if you are using sgd with your typical CNN architecture
# sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)

# calling the compile function for the ResNet-34
cnn_model.compile(
    loss='binary_crossentropy',
    optimizer=tf.optimizers.Nadam(),
    metrics=['accuracy'])

epochs=30

# Training
import time
start_time = time.time()

history = cnn_model.fit(x_train, y_train, batch_size=32, epochs=epochs, validation_split=0.3, shuffle=True, callbacks=[cp_callback])

training_time = time.time() - start_time

!ls {checkpoint_dir}
#cnn_model.load_weights(checkpoint_path)

#Evaluating model on testing set
cnn_model.evaluate(x_test, y_test)

classifications = cnn_model.predict(x_test)

#Visualizing Loss and Accuracy

mm = training_time // 60
ss = training_time % 60
print('Training {} epochs in {}:{}'.format(epochs, int(mm), round(ss, 1)))

# show the loss and accuracy
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# loss plot
plt.plot(loss)
plt.plot(val_loss, 'r')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend(["Training", "Validation"])

plt.show()

# accuracy plot
plt.plot(acc)
plt.plot(val_acc, 'r')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend(['Training', 'Validation'], loc=4)

