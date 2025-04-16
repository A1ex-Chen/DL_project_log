def __init__(self, d):
    marker = eval(d.argMarker[0])
    mod = marker['mod']
    op = marker['op']
    args = marker['args']
    self.marker = marker
    self.mod_ = mod
    self.op_ = op
    self.args = args
    assert op == 'adam'
    assert len(args) == 12 or len(args) == 14
    w, hw, m, v, g = args[0:5]
    assert w['shape'] == m['shape'] == v['shape'] == g['shape']
    assert hw['shape'] == w['shape'] or hw['shape'] == (0,)
    assert w['type'] == m['type'] == v['type'] == g['type'] == hw['type'
        ] == 'tensor'
    assert w['dtype'] == m['dtype'] == v['dtype'] == 'float32'
    self.w = w
    self.g = g
