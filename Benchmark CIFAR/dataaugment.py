import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import random
class processor():

    def __init__(self, image, randpatch, randcontrast, randbrightness):
        self.image = image
        self.randpatch = randpatch
        self.randcontrast = randcontrast
        self.randbrightness = randbrightness

    def augment(image):
        randpatch = random.randint(-4,4)
        image = image[int(randpatch):(int(randpatch)+24),int(randpatch):(int(randpatch)+24),:]
        flipped = tf.image.flip_left_right(image)
        brightness = tf.image.random_brightness(flipped)
        contrast = tf.image.random_contrast(brightness)
        return contrast



