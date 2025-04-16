def _dot_product(xs, ys):
    """Calculate the vector inner product between two lists of Tensors."""
    return sum([(x * y) for x, y in zip(xs, ys)])
