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
    assert op == 'any'
    assert len(args) == 1
    t = args[0]
    self.shape = t['shape']
    self.type = t['dtype']
    self.sub = d.sub
    return
