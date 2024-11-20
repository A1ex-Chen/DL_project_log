def bmm(node):
    b, n, m = node.inputs[0].shape
    b, m, p = node.inputs[1].shape
    return b * n * m * p
