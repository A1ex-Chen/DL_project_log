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
    assert op == 'log_softmax'
    args = list(filter(lambda x: x['name'] == '', args))
    assert len(args) <= 2
    if args[0]['name'] == '':
        i = args[0]
    else:
        i = list(filter(lambda x: x['name'] == 'input', args))[0]
    t = i['dtype']
    self.shape = i['shape']
    self.type = i['dtype']
    self.dir = d.dir
    return
