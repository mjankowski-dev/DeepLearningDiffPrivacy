from momentacc import *
from dp_sgd import *
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import matplotlib.ticker as mticker
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.linalg import eigh
from tqdm import tqdm

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28), 'Shape not equal'
assert x_test.shape == (10000, 28, 28), 'Shape not equal'
assert y_train.shape == (60000,), 'Shape not equal'
assert y_test.shape == (10000,), 'Shape not equal'
print('Dataset booted')



x_test = x_test.reshape(x_test.shape[0], 784)
x_train = x_train.reshape(x_train.shape[0],
                          784)
totalset = np.concatenate((x_test, x_train))
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
totalset = scaler.fit_transform(totalset)


covar = np.matmul(totalset.T, totalset)  # Compute the covariance matrix on the data set
shape = tf.shape(covar)
noise = tf.random.normal(shape, mean=0, stddev=16, dtype=tf.dtypes.float32)  # Create noise matrix of the same shape
noise = (noise + noise.T) / 2  # Make sure noise matrix is symmetric
covar += noise  # Add the noise
def pca(data, covar):
    values, vectors = eigh(covar, eigvals=(724, 783))  #Calculate vectors of covariance matrix
    vectors = vectors.T
    final_data = np.matmul(vectors, data.T)  # Apply projection to data

    return final_data


x_test = pca(x_test, covar).T
x_train = pca(x_train, covar).T

indices = np.arange(0, len(x_train))
np.random.shuffle(indices)   # Randomizing the batching
batches = 100
x_train_batches = np.array_split(x_train[indices, :], batches)  # Creating the batches from the training set
y_train_batches = np.array_split(y_train[indices], batches)
train = list(zip(x_train_batches, y_train_batches))
x_test_batches = np.array_split(x_test, batches)  # Creating the batches from the test set
y_test_batches = np.array_split(y_test, batches)
test = list(zip(x_test_batches, y_test_batches))

seed = 2
std_pca = 16  # std for pca
std_sgd = 8  # std for dp_sgd
batch_size = 600
lr_sgd = 0.05  # [0.01,0.07] stable, best at 0.05
C = 4  # gradient clipping bound
gs = batch_size
np.random.seed(seed)

parameters_ma = {"maxOrder": 32,
                 "sigma": std_sgd,
                 "q": batch_size / 60000,
                 "T": 400}
debug = True
# '''
deltaFixed = False
epsFixed = True
epsilon = 0.5
th_delta = 100  # 10**-5 # epsilon fixed
'''
deltaFixed = True 
epsFixed= False
delta = 10e-5
th_epsilon = 2 # delta fixed
'''
allParameters = {**parameters_ma,
                 'seed': seed,
                 'std_pca': std_pca,
                 'std_sgd': std_sgd,
                 'batch_size': batch_size,
                 'lr_sgd': lr_sgd,
                 'C': C,
                 'deltaFixed': deltaFixed,
                 'epsFixed': epsFixed,
                 'epsilon': epsilon,
                 'th_delta': th_delta,
                 }

optimizer = tf.keras.optimizers.SGD(learning_rate=lr_sgd)  # [0.01,0.07] stable, best at 0.05
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.compat.v1.losses.Reduction.NONE)
metric_train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
metric_val_acc = tf.keras.metrics.SparseCategoricalAccuracy()


def train_epoch(dp_sgd, optimizer, model, loss_object, moment_accountant):
    losses = []
    progress = tqdm(total=len(list(enumerate(train))), position=0, leave=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ')  # progress bar
    go = True
    step = 0
    for x_batch_train, y_batch_train in train:  #Applying gradients and calculating moments for each batch
        step += 1
        logit, loss = dp_sgd.apply_gradients(optimizer, model, loss_object, x_batch_train, y_batch_train)
        delta, eps = moment_accountant.compute_deltaEps()

        losses.append(loss)   # Creating loss list

        metric_train_acc(y_batch_train, logit)
        progress.set_description("Train loss for step number%s: %.4f" % (int(step), float(loss)))
        progress.update()

        if not moment_accountant.check_thresholds(delta, epsilon):   # Privacy budget threshold
            go = False
            break
    print(f'Delta = {delta} | Epsilon = {epsilon}')
    return losses, go


def validate():
    losses = []
    for x_val, y_val in test:
        logits_val = model(x_val)
        loss_val = loss_object(y_val, logits_val)
        losses.append(loss_val)
        metric_val_acc(y_val, logits_val)
    return losses


def build_model():
    inputs = tf.keras.Input(shape=(60,), name='digits')
    inputs = DP_PCA(60, seed, std_pca)(inputs)
    x = tf.keras.layers.Dense(1000, activation='relu', name='dense_1')(inputs)
    outputs = tf.keras.layers.Dense(10, activation='softmax', name='predictions')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model


## INITIALIZE
model = build_model()
model.layers[2].trainable = False  # pca layer
DPSGD = DP_SGD(lr_sgd, std_sgd, gs, C, seed)
if epsFixed:
    print("\n Epsilon kept fixed \n")
    accountant = moment_accountant(seed, parameters_ma, deltaFixed=deltaFixed, epsFixed=epsFixed, debug=debug,
                                   epsilon=epsilon, th_delta=th_delta)
else:
    print("\n Delta kept fixed \n")
    # delta fixed
    accountant = moment_accountant(seed, parameters_ma, deltaFixed=deltaFixed, epsFixed=epsFixed, debug=debug,
                                   delta=delta, th_epsilon=th_epsilon)

# Iterate over epochs.
epochs = 50  # 18
val_acc_list, train_acc_list = [], []
for epoch in range(epochs):
    print('Start of epoch %d' % (epoch,))

    train_losses, go = train_epoch(DPSGD, optimizer, model, loss_object, accountant)
    train_acc = metric_train_acc.result()

    val_losses = validate()
    val_acc = metric_val_acc.result()

    val_acc_list.append(val_acc)
    train_acc_list.append(train_acc)

    print('\n Epoch %s:, Training Accuracy: %.4f, Test Accuracy %.4f' % (
    epoch, float(train_acc), float(val_acc)))

    metric_train_acc.reset_states()
    metric_val_acc.reset_states()
    if not go:
        print(f"\n Stopping due to privacy loss at epoch {epoch}/{epochs} \n")
        break


