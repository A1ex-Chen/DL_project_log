def __init__(self, d):
    marker = eval(d.argMarker[0])
    mod = marker['mod']
    op = marker['op']
    args = marker['args']
    self.marker = marker
    self.mod_ = mod
    self.op_ = op
    self.args = args
    assert mod == 'torch'
    assert op == 'randperm'
    assert len(args) == 1
    n = args[0]
    assert n['type'] == 'int'
    self.n = n['value']