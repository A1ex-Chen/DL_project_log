def gelu(x):
    """ Implementation of the gelu activation function.
        XLNet is using OpenAI GPT's gelu
        Also see https://arxiv.org/abs/1606.08415
    """
    cdf = 0.5 * (1.0 + tf.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(
        x, 3))))
    return x * cdf
