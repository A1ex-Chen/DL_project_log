def gelu_fast(x):
    x = tf.convert_to_tensor(x)
    coeff1 = tf.cast(7978845608, x.dtype)
    coeff2 = tf.cast(0.044715, x.dtype)
    return 0.5 * x * (1.0 + tf.tanh(x * coeff2 * (1.0 + coeff1 * x * x)))
