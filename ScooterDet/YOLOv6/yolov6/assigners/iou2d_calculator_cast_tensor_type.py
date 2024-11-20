def cast_tensor_type(x, scale=1.0, dtype=None):
    if dtype == 'fp16':
        x = (x / scale).half()
    return x
