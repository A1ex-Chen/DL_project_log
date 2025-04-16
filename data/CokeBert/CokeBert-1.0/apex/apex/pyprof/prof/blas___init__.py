def __init__(self, d):
    marker = eval(d.argMarker[0])
    mod = marker['mod']
    op = marker['op']
    args = marker['args']
    self.marker = marker
    self.mod_ = mod
    self.op_ = op
    self.args = args
    assert mod == 'torch' and op == 'mm'
    assert len(args) == 2
    A, B = args
    m, k1 = A['shape']
    k2, n = B['shape']
    assert k1 == k2
    t1 = A['dtype']
    t2 = B['dtype']
    assert t1 == t2
    self.A = A
    self.B = B
    self.m = m
    self.n = n
    self.k = k1
    self.type = t1
    self.name = d.name
    return
