def __init__(self, d):
    marker = eval(d.argMarker[0])
    mod = marker['mod']
    op = marker['op']
    args = marker['args']
    self.marker = marker
    self.mod_ = mod
    self.op_ = op
    self.args = args
    self.sub = d.sub
    assert mod == 'Tensor' or mod == 'torch'
    assert op == 'masked_select'
    args = list(filter(lambda x: x['name'] != 'out', args))
    assert len(args) == 2
    if args[0]['name'] == '':
        t = args[0]
    else:
        t = list(filter(lambda x: x['name'] == 'input', args))[0]
    if args[1]['name'] == '':
        m = args[1]
    else:
        m = list(filter(lambda x: x['name'] == 'mask', args))[0]
    assert m['dtype'] == 'uint8'
    tensor = t['shape']
    mask = m['shape']
    if tensor != mask:
        array1 = np.empty(list(tensor))
        array2 = np.empty(list(mask))
        try:
            out = np.broadcast(array1, array2).shape
        except:
            assert False
    self.tshape = tensor
    self.mshape = mask
    self.type = t['dtype']
