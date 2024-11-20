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
    assert op == 'mse_loss'
    assert len(args) == 3
    if args[0]['name'] == '':
        x = args[0]
    else:
        x = list(filter(lambda x: x['name'] == 'input', args))[0]
    if args[1]['name'] == '':
        y = args[1]
    else:
        y = list(filter(lambda x: x['name'] == 'target', args))[0]
    if args[2]['name'] == '':
        r = args[2]
    else:
        r = list(filter(lambda x: x['name'] == 'reduction', args))[0]
    assert x['type'] == y['type'] == 'tensor'
    assert x['shape'] == y['shape']
    assert x['dtype'] == y['dtype']
    assert r['type'] == 'str'
    assert r['value'] in ['none', 'mean', 'sum']
    self.shape = x['shape']
    self.type = x['dtype']
    self.red = r['value']
    self.dir = d.dir
