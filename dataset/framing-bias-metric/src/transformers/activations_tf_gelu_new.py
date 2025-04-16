def gelu_new(x):
    """
    Gaussian Error Linear Unit. This is a smoother version of the GELU. Original paper: https://arxiv.org/abs/1606.0841

    Args:
        x: float Tensor to perform activation

    Returns:
        `x` with the GELU activation applied.
    """
    x = tf.convert_to_tensor(x)
    pi = tf.cast(math.pi, x.dtype)
    coeff = tf.cast(0.044715, x.dtype)
    cdf = 0.5 * (1.0 + tf.tanh(tf.sqrt(2.0 / pi) * (x + coeff * tf.pow(x, 3))))
    return x * cdf
