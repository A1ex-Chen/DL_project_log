def matmul(node):
    if node.inputs[0].ndim == 1 and node.inputs[1].ndim == 1:
        n = node.inputs[0].shape[0]
        return n
    if node.inputs[0].ndim == 1 and node.inputs[1].ndim == 2:
        n, m = node.inputs[1].shape
        return n * m
    if node.inputs[0].ndim == 2 and node.inputs[1].ndim == 1:
        n, m = node.inputs[0].shape
        return n * m
    if node.inputs[0].ndim == 2 and node.inputs[1].ndim == 2:
        n, m = node.inputs[0].shape
        m, p = node.inputs[1].shape
        return n * m * p
    if node.inputs[0].ndim == 1:
        *b, n, m = node.inputs[1].shape
        return prod(b) * n * m
    if node.inputs[1].ndim == 1:
        *b, n, m = node.inputs[0].shape
        return prod(b) * n * m
    *b, n, p = node.outputs[0].shape
    *_, n, m = node.inputs[0].shape
    *_, m, p = node.inputs[1].shape
    return prod(b) * n * m * p
