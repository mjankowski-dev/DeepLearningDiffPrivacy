import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


y_train, y_test = y_train.flatten(), y_test.flatten()
y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
y_test = tf.one_hot(y_test.astype(np.int32), depth=10)

#train_images = train_images.reshape(50000,3,32,32).transpose(0,2,3,1)
#test_images = test_images.reshape(10000,3,32,32).transpose(0,2,3,1)
#y_train = y_train.flatten()
#y_test = y_test.flatten()
#y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
#y_test = tf.one_hot(y_test.astype(np.int32), depth=10)

x_train = x_train[:,4:28,4:28,:]
x_test = x_test[:,4:28,4:28,:]
#x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
#x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
#print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


def preprocess(image):
    image = image[4:28,4:28]
    image = image / 255.0
    return image

x_train_processed = []
x_test_processed = []
#for image in x_train:
#    x_train_processed.append(preprocess(image))

#for image in x_test:
#    x_test_processed.append((preprocess(image)))
#print(y_train)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(64,(5, 5),  activation='relu', input_shape=(24, 24, 3), ))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64,(5, 5), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(384, activation='relu'))
model.add(tf.keras.layers.Dense(384, activation='relu'))
model.add(tf.keras.layers.Dense(10))
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(train_images, train_labels, batch_size=50,
                    epochs=100 )



