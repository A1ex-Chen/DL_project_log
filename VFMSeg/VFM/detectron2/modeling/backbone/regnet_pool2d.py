def pool2d(k, *, stride=1):
    """Helper for building a pool2d layer."""
    assert k % 2 == 1, 'Only odd size kernels supported to avoid padding issues.'
    return nn.MaxPool2d(k, stride=stride, padding=(k - 1) // 2)
