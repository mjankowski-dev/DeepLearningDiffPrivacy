import tensorflow as tf
#import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Normalization

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from tqdm import tqdm # gives progress bar when loading


import time
from sklearn.preprocessing import StandardScaler
import random

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.compat.v1.losses.Reduction.NONE)
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

def apply_gradient(optimizer, model, x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss_value = loss_object(y_true=y, y_pred=logits)

    gradients = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights)) # ZIP ENSURES THAT GRADIENTS IS APPLIED TO EVERY LAYER CORRECTLY (SINCE EVERY LAYER HAS W & b != not 1 variable)

    return logits, loss_value




def perform_validation():
    losses = []
    for x_val, y_val in test:
        val_logits = model(x_val)
        val_loss = loss_object(y_val, val_logits)
        #val_loss = tf.reduce_sum(val_loss, axis = 0).numpy()
        losses.append(val_loss)
        val_acc_metric(y_val, val_logits)
    return losses


