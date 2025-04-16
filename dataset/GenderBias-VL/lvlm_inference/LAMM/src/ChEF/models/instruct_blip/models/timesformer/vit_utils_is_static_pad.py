def is_static_pad(kernel_size: int, stride: int=1, dilation: int=1, **_):
    return stride == 1 and dilation * (kernel_size - 1) % 2 == 0
