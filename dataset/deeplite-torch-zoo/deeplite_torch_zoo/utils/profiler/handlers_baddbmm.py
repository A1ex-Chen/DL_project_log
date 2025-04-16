def baddbmm(node):
    b, n, p = node.inputs[0].shape
    b, n1, m = node.inputs[1].shape
    b, m1, p1 = node.inputs[2].shape
    assert n == n1 and m == m1 and p == p1
    return b * n * m * p
