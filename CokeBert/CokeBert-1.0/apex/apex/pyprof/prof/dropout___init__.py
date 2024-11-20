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
    assert op == 'dropout'
    self.shape = args[0]['shape']
    self.type = args[0]['dtype']
    self.dir = d.dir
    return
