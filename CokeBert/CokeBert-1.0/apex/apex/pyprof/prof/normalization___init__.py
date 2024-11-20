def __init__(self, d):
    marker = eval(d.argMarker[0])
    mod = marker['mod']
    op = marker['op']
    args = marker['args']
    self.marker = marker
    self.mod_ = mod
    self.op_ = op
    self.args = args
    assert op == 'batch_norm'
    assert len(args) == 8
    i = args[0]
    assert i['type'] == 'tensor'
    self.shape = i['shape']
    self.type = i['dtype']
    self.dir = d.dir
