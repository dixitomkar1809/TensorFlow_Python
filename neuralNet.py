# Training first neural network using tensor flow
# Source: https://www.tensorflow.org/tutorials/keras/basic_classification
# Author: Omkar Dixit

# Tensorflow and tf.keras
import tensorflow as tf
from tensorflow import keras

#Helper Libraries
import numpy as np
import matplotlib.pyplot as plt

# print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(training_data, training_labels), (testing_data, testing_labels) = fashion_mnist.load_data()

# Since the data doesn't describe the class labels, we store them separately
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Exploring the data
# print(training_data.shape)
# print(len(training_labels))
# print(testing_data.shape)
# print(len(testing_labels))

# Data Preprocessing
plt.interactive(False)
plt.figure()
plt.imshow(training_data[0])
plt.colorbar()
#plt.show()
training_data = training_data / 255.0
testing_data = testing_data / 255.0

# Displaying first few images to make sure that correct data is being trained
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(training_data[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i]])
#plt.show()

# Building the Neural Network
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(150, activation=tf.nn.relu),
    keras.layers.Dense(150, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compiling the Model
model.compile(optimizer=tf.train.AdadeltaOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(training_data, training_labels, epochs=5)

# Evaluate the accuracy
testing_loss, testing_accuracy = model.evaluate(testing_data, testing_labels)
print("Test Accuracy: ", testing_accuracy)

# Prediction
predictions = model.predict(testing_data)

# Printing the first prediction's confidence array
print(predictions[0])

# Class label with max confidence
#print(np.argmax(predictions[0]))

# Printing the test label to make sure they are the same
# print(testing_labels[0])

# Plotting Graphs
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color='red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100 * np.max(predictions_array), class_names[true_label]), color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# taking a look at one specific image

# i = 0
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions, testing_labels, testing_data)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions, testing_labels)
#
# plt.show()

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, testing_labels, testing_data)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, testing_labels)
#plt.show()

# Grab an image from the test dataset
img = testing_data[0]
print(img.shape)

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))
print(img.shape)

predictions_single = model.predict(img)
print(predictions_single)

plot_value_array(0, predictions_single, testing_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

np.argmax(predictions_single[0])

plt.show()

#@title MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.