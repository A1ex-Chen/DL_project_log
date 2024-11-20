@staticmethod
def _separate_heads(x: Tensor, num_heads: int) ->Tensor:
    """Separate the input tensor into the specified number of attention heads."""
    b, n, c = x.shape
    x = x.reshape(b, n, num_heads, c // num_heads)
    return x.transpose(1, 2)
