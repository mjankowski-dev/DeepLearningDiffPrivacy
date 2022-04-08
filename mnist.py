
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
from scipy.linalg import eigh

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28), 'Shape not equal'
assert x_test.shape == (10000, 28, 28), 'Shape not equal'
assert y_train.shape == (60000,), 'Shape not equal'
assert y_test.shape == (10000,), 'Shape not equal'
print('Dataset booted')
#Each example is a 28 × 28 size gray-level image. We use a simple feedforward neural network with ReLU units and softmax of 10
#classes (corresponding to the 10 digits) with cross-entropy
#loss and an optional PCA input layer.
#Baseline model.
#Our baseline model uses a 60-dimensional PCA projection
#layer and a single hidden layer with 1,000 hidden units. Using the lot size of 600, we can reach accuracy of 98.30% in
#about 100 epochs
x_train = x_train.reshape(x_train.shape[0],
                         784)
scaler = StandardScaler()
x_train= scaler.fit_transform(x_train)
#print(std_x.shape)
#pca = PCA(n_components=60)
#x_train = pca.fit_transform(std_x)
#print(x_train.shape)
y_train = tf.keras.utils.to_categorical(y_train)




def base_model():
    inputs = tf.keras.Input(shape=(60,), name='digits')
    x = tf.keras.layers.Dense(1000, activation='relu', name='dense_1')(inputs)
    outputs = tf.keras.layers.Dense(10, activation='softmax', name='predictions')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def pca(data):
    print('progress 1')
    covar_matrix = np.matmul(data.T, data)
    print ( "The shape of variance matrix = ", covar_matrix.shape)
    values, vectors = eigh(covar_matrix, eigvals=(724,783))

    vectors = vectors.T
    print(vectors.shape)
    final_data = np.matmul(vectors, data.T)

    return final_data
x_train = [pca(x) for x in x_train]

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1000, input_dim=60, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['categorical_accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=600, validation_split=0.15, verbose=2)