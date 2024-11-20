def setXWBMNK(self, args):
    x = None
    w = None
    b = None
    if len(args) == 2:
        x, w = args
    elif len(args) == 3:
        x, w, b = args
        assert x['type'] == w['type'] == 'tensor'
        if b['type'] == 'tensor':
            assert len(b['shape']) == 1
        elif b['type'] == 'NoneType':
            assert b['value'] is None
            b = None
        else:
            assert False
    else:
        assert False
    assert len(w['shape']) == 2
    k1 = x['shape'][-1]
    n, k2 = w['shape']
    assert k1 == k2
    if b is not None:
        assert b['shape'][0] == n
    t1 = x['dtype']
    t2 = w['dtype']
    assert t1 == t2
    self.x = x['shape']
    self.w = w['shape']
    self.b = b['shape'] if b is not None else None
    self.type = t1
    n = self.x[0:-1]
    k = self.x[-1]
    m, k1 = self.w
    assert k == k1
    self.m = m
    self.n = n
    self.k = k
