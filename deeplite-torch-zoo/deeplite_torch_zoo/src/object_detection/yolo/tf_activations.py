def activations(act=nn.SiLU):
    if isinstance(act, nn.ReLU):
        return lambda x: keras.activations.relu(x)
    if isinstance(act, nn.LeakyReLU):
        return lambda x: keras.activations.relu(x, alpha=0.1)
    elif isinstance(act, nn.Hardswish):
        return lambda x: x * tf.nn.relu6(x + 3) * 0.166666667
    elif isinstance(act, (nn.SiLU, SiLU)):
        return lambda x: keras.activations.swish(x)
    else:
        raise Exception(
            f'no matching TensorFlow activation found for PyTorch activation {act}'
            )
