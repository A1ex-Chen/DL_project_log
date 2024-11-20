def _scaled_dot_product(scale, xs, ys):
    """Calculate a scaled, vector inner product between lists of Tensors."""
    return sum([(scale * x * y) for x, y in zip(xs, ys) if 
        _possibly_nonzero(x) or _possibly_nonzero(y)])
