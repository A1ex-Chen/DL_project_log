def __init__(self, d):
    marker = eval(d.argMarker[0])
    mod = marker['mod']
    op = marker['op']
    args = marker['args']
    self.marker = marker
    self.mod_ = mod
    self.op_ = op
    self.args = args
    self.name = d.name
    self.dir = d.dir
    self.sub = d.sub
    self.grid = d.grid
    assert op == 'forward'
    assert mod in ['LSTMCell', 'GRUCell', 'RNNCell']
    assert len(args) in [2, 3]
    x, h = args[0], args[1]
    b1, ii = x['shape']
    b2, hh = h['shape']
    assert b1 == b2
    assert x['dtype'] == h['dtype']
    t = x['dtype']
    self.cell = mod
    self.inp = ii
    self.hid = hh
    self.b = b1
    self.type = t
    self.multiple = 1
    if self.cell == 'LSTMCell':
        self.multiple = 4
    elif self.cell == 'GRUCell':
        self.multiple = 3
    self.gemm = None
    self.m = None
    self.n = None
    self.k = None
    self.elems = 0
    self.bar()
