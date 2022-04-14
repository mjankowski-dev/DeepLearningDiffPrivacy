import tensorflow as tf


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.compat.v1.losses.Reduction.NONE)
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()


class DP_SGD:
    def __init__(self, lr=0.01, sigma=0.2, gs=10, C=1, seed=2):  # TODO; check defaults
        self.lr = lr  # learning rate
        self.sigma = sigma  # sigma, noise scale
        self.gs = gs  # group size
        self.C = C  # gradient norm bound
        self.seed = seed
        self.std = sigma * C  # for noise addition


    def apply_gradients(self, optimizer, model, loss_object, x, y):


        with tf.GradientTape(persistent=True) as tape:
            y_pred = model(x)
            loss = loss_object(y, y_pred)
            loss_red = tf.reduce_sum(loss, axis=0)


        grad = tape.jacobian(loss, model.trainable_weights, parallel_iterations=None, experimental_use_pfor=False)

        ## clip gradients per layer
        for l in range(len(grad)):
            # clipper = tf.norm(grad[l], ord = 2, axis = 0)
            dims = len(tf.shape(grad[l]))
            clipper = tf.math.square(grad[l])

            if dims > 2:
                # kernel layers
                clipper = tf.reduce_sum(clipper, axis=[1, 2])
            else:
                # bias layer
                clipper = tf.reduce_sum(clipper, axis=-1)

            clipper = tf.math.sqrt(clipper)
            clipper = tf.math.maximum(tf.constant([1], dtype=tf.dtypes.float32), clipper / self.C)
            if dims > 2:
                # kernel layers
                clipper = tf.broadcast_to(tf.expand_dims(tf.expand_dims(clipper, -1), -1), tf.shape(grad[l]))
            else:
                # bias layer
                clipper = tf.broadcast_to(tf.expand_dims(clipper, -1), tf.shape(grad[l]))
            grad[l] = tf.math.divide(grad[l], clipper)  # override

        ## add noise
        for l in range(len(grad)):  # loop over layers
            grad_red = tf.math.reduce_sum(grad[l], axis=0)
            shape = tf.shape(grad_red)
            noise = tf.random.normal(shape, mean=0, stddev=self.std, dtype=tf.dtypes.float32, seed=self.seed)
            grad[l] = tf.add(grad_red, noise) / self.gs



        optimizer.apply_gradients(zip(grad, model.trainable_weights))

        ## collect

        return y_pred, loss_red

