@staticmethod
def _recombine_heads(x: Tensor) ->Tensor:
    """Recombine the separated attention heads into a single tensor."""
    b, n_heads, n_tokens, c_per_head = x.shape
    x = x.transpose(1, 2)
    return x.reshape(b, n_tokens, n_heads * c_per_head)
