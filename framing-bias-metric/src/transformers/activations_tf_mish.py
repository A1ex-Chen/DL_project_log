def mish(x):
    x = tf.convert_to_tensor(x)
    return x * tf.tanh(tf.math.softplus(x))
