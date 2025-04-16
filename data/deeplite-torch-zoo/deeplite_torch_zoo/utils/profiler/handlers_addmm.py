def addmm(node):
    n, m = node.inputs[1].shape
    m, p = node.inputs[2].shape
    return n * m * p
