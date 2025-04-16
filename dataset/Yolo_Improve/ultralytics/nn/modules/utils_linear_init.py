def linear_init(module):
    """Initialize the weights and biases of a linear module."""
    bound = 1 / math.sqrt(module.weight.shape[0])
    uniform_(module.weight, -bound, bound)
    if hasattr(module, 'bias') and module.bias is not None:
        uniform_(module.bias, -bound, bound)
