# Python Machine Learning tutorial: Neural Networks using Keras
# Source: https://www.tensorflow.org/tutorials/keras/classification?hl=pt-br
# Kaique Guimarães Cerqueira

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
#from threading import Event
import keyboard
import sys

# Loag dataset
mnist = keras.datasets.fashion_mnist
# Load data separated between train and test into numPy arrays
(train_img, train_label), (test_img, test_label) = mnist.load_data()
# Normalize data
train_img = train_img / 255.0
test_img = test_img / 255.0

# Classification names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Set up model of neural net
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)), # 28*28 = 784 neurons at input layer
  tf.keras.layers.Dense(128, activation='relu'),
  #tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(train_img, train_label, epochs=5)

# Use test data to evaluate
print('\n-------------------------------------------------\n')
loss, accuracy = model.evaluate(test_img, test_label, verbose=1)

# Predict the test data
predictions = model.predict(test_img)
print(predictions[0])
# Plot predictions
iter = 0
running = True

plt.figure()
while running:

  plt.subplot(1,2,1)
  plt.imshow(test_img[iter], cmap=plt.cm.binary)
  plt.title('Prediction No. '+ str(iter+1))
  predicted_label = np.argmax(predictions[iter])
  plt.xlabel('Predicted: {} ({:2.0f}%)   Real: {}'.format(class_names[predicted_label],
                                                          100 * np.max(predictions[iter]),
                                                          class_names[test_label[iter]]))

  plt.subplot(1,2,2)
  plt.bar(range(len(class_names)), predictions[iter])
  plt.ylim(0,1)
  plt.xticks(range(len(class_names)), class_names, rotation=45)

  plt.show()
  iter += 1
