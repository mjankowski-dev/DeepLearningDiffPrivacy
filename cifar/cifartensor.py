import tensorflow as tf
import random
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
train_images = train_images[:,:,:,:]

train_images = tf.convert_to_tensor(train_images, np.float32)
#train_images = train_images[:,4:28,4:28,:]
#test_images = test_images[:,4:28,4:28,:]

print(train_images.shape)


def augment(image):
    #image = np.reshape(image,(32,32,3))
    #print(image.shape)

    randpatch = random.randint(0, 8)
    image = image[int(randpatch):(int(randpatch) + 24), int(randpatch):(int(randpatch) + 24), :]
    flipped = tf.image.flip_left_right(image)
    brightness = tf.image.random_brightness(flipped, max_delta=0.95)
    contrast = tf.image.random_contrast(brightness, lower=0.1, upper=0.9)
    #print(contrast.shape)
    return contrast

train_images1 = np.array([augment(xi) for xi in train_images])
test_images = np.array([augment(xi) for xi in test_images])
#dataset = train_images.map(augment)
print('Augmentation finished')
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


model = models.Sequential()
model.add(layers.Conv2D(64,(5, 5), padding = 'same', input_shape=(24, 24, 3), data_format="channels_last"))
model.add(layers.Dropout(0.25))
model.add(layers.ReLU())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64,(5, 5), padding= 'same'))
model.add(layers.Dropout(0.25))
model.add(layers.ReLU())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(384, activation='relu'))
model.add(layers.Dense(384, activation='relu'))
model.add(layers.Dense(10))

model.summary()

model.compile(optimizer='SGD', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = model.fit(train_images1, train_labels, epochs=500, batch_size=300, validation_data=(test_images, test_labels))