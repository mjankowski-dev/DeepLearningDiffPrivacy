import tensorflow as tf

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


