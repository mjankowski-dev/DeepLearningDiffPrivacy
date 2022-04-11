from momentacc import *
from trainingloop import *
from dp_sgd import *
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import matplotlib.ticker as mticker
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.linalg import eigh

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
    values, vectors = eigh(covar, eigvals=(724, 783))
    vectors = vectors.T
    final_data = np.matmul(vectors, data.T)

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
std_pca = 16  # 4 # std for pca
std_sgd = 8  # std for dp_sgd
batch_size = 600
lr_sgd = 0.05  # [0.01,0.07] stable, best at 0.05
C = 4  # gradient clipping bound
gs = batch_size
np.random.seed(seed)

# moment accountant specific
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
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()


def train_data_for_one_epoch(dp_sgd, optimizer, model, loss_object, moment_accountant):
    losses = []
    pbar = tqdm(total=len(list(enumerate(train))), position=0, leave=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ')  # progress bar
    go = True
    step = 0
    for x_batch_train, y_batch_train in train:  #Applying gradients and calculating moments for each batch
        step += 1
        logits, loss_value = dp_sgd.apply_gradients(optimizer, model, loss_object, x_batch_train, y_batch_train)
        delta, eps = moment_accountant.compute_deltaEps()

        losses.append(loss_value)   # Creating loss list

        train_acc_metric(y_batch_train, logits)
        pbar.set_description("Training loss for step %s: %.4f" % (int(step), float(loss_value)))
        pbar.update()

        if not moment_accountant.check_thresholds(delta, epsilon):   # Privacy budget threshold
            go = False
            break
     print(f'Delta = {delta} | Epsilon = {epsilon}')
    return losses, go


def perform_validation():
    losses = []
    for x_val, y_val in test:
        val_logits = model(x_val)
        val_loss = loss_object(y_val, val_logits)
        # val_loss = tf.reduce_sum(val_loss, axis = 0).numpy()
        losses.append(val_loss)
        val_acc_metric(y_val, val_logits)
    return losses


def base_model():
    inputs = tf.keras.Input(shape=(60,), name='digits')
    x = DP_PCA(60, seed, std_pca)(inputs)
    x = tf.keras.layers.Dense(1000, activation='relu', name='dense_1')(inputs)
    outputs = tf.keras.layers.Dense(10, activation='softmax', name='predictions')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model


## INITIALIZE
model = base_model()
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
epochs_val_losses, epochs_train_losses = [], []
for epoch in range(epochs):
    print('Start of epoch %d' % (epoch,))

    losses_train, go = train_data_for_one_epoch(DPSGD, optimizer, model, loss_object, accountant)
    train_acc = train_acc_metric.result()

    losses_val = perform_validation()
    val_acc = val_acc_metric.result()

    losses_train_mean = np.mean(losses_train)
    losses_val_mean = np.mean(losses_val)
    epochs_val_losses.append(losses_val_mean)
    epochs_train_losses.append(losses_train_mean)

    print('\n Epoch %s: Train loss: %.4f  Validation Loss: %.4f, Train Accuracy: %.4f, Validation Accuracy %.4f' % (
    epoch, float(losses_train_mean), float(losses_val_mean), float(train_acc), float(val_acc)))

    train_acc_metric.reset_states()
    val_acc_metric.reset_states()
    if not go:
        print(f"\n Stopping due to privacy loss at epoch {epoch}/{epochs} \n")
        break


def plot_metrics(train_metric, val_metric, metric_name, title, ylim=5):
    plt.title(title)
    # plt.ylim(0,ylim)
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    x = np.arange(1, len(train_metric) + 1)
    plt.plot(x, train_metric, color='blue', label=metric_name)
    plt.plot(x, val_metric, color='green', label='val_' + metric_name)
    plt.legend()


plot_metrics(epochs_train_losses, epochs_val_losses, "Loss", "Loss", ylim=10.0)
