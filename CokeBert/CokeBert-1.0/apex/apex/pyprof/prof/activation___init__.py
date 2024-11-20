def __init__(self, d):
    marker = eval(d.argMarker[0])
    mod = marker['mod']
    op = marker['op']
    args = marker['args']
    self.marker = marker
    self.mod_ = mod
    self.op_ = op
    self.args = args
    assert mod in ['torch.nn.functional', 'torch', 'Tensor']
    args = list(filter(lambda x: x['name'] == '', args))
    assert len(args) >= 1
    arg = args[0]
    assert arg['type'] == 'tensor'
    self.i = arg
    self.dir = d.dir
