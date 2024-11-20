def __init__(self, d):
    marker = eval(d.argMarker[0])
    mod = marker['mod']
    op = marker['op']
    args = marker['args']
    self.marker = marker
    self.mod_ = mod
    self.op_ = op
    self.args = args
    assert mod == 'torch.nn.functional'
    assert op == 'embedding'
    self.ishape = args[0]['shape']
    self.itype = args[0]['dtype']
    self.eshape = args[1]['shape']
    self.etype = args[1]['dtype']
    assert len(self.eshape) == 2
    self.dir = d.dir
    self.sub = d.sub
    return
