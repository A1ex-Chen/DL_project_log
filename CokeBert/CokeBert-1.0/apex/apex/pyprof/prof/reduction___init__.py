def __init__(self, d):
    marker = eval(d.argMarker[0])
    mod = marker['mod']
    op = marker['op']
    args = marker['args']
    self.marker = marker
    self.mod_ = mod
    self.op_ = op
    self.args = args
    assert mod in ['torch', 'Tensor']
    assert op == 'norm'
    i = args[0]
    self.shape = i['shape']
    self.type = i['dtype']
