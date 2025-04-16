def __init__(self, d):
    marker = eval(d.argMarker[0])
    mod = marker['mod']
    op = marker['op']
    args = marker['args']
    self.marker = marker
    self.mod_ = mod
    self.op_ = op
    self.args = args
    assert mod == 'Tensor'
    assert op in Convert.ops
    assert len(args) == 1
    t = args[0]
    if t['type'] == 'tensor':
        shape = t['shape']
        stype = t['dtype']
    else:
        shape = 1,
        stype = t['type']
    if self.op_ == 'to':
        op = stype
    self.shape = shape
    self.stype = stype
    self.dtype = op
