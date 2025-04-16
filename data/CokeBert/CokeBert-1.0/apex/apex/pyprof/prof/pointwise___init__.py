def __init__(self, d):
    marker = eval(d.argMarker[0])
    mod = marker['mod']
    op = marker['op']
    args = marker['args']
    self.marker = marker
    self.mod_ = mod
    self.op_ = op
    self.args = args
    self.dir = d.dir
    assert d.dir in ['fprop', 'bprop']
    assert op in Pointwise.ops
    args = list(filter(lambda x: x['name'] == '', args))
    args = list(filter(lambda x: x['type'] == 'tensor', args))
    if len(args) == 0:
        self.shape = [(1,)]
        self.type = 'float32'
    elif len(args) == 1:
        in0 = args[0]
        _, t0, s0, dt0 = Pointwise.foo(in0)
        assert t0 == 'tensor'
        self.shape = [s0]
        self.type = dt0
    elif len(args) == 2:
        in0, in1 = args
        _, t0, s0, dt0 = Pointwise.foo(in0)
        _, t1, s1, dt1 = Pointwise.foo(in1)
        assert t0 == t1 == 'tensor'
        assert dt0 == dt1
        self.shape = [s0, s1]
        self.type = dt0
    elif len(args) == 3:
        in0, in1, in2 = args
        _, t0, s0, dt0 = Pointwise.foo(in0)
        _, t1, s1, dt1 = Pointwise.foo(in1)
        _, t2, s2, dt2 = Pointwise.foo(in2)
        assert t0 == t1 == t2 == 'tensor'
        assert dt0 == dt1 == dt2
        self.shape = [s0, s1, s2]
        self.type = dt0
    else:
        assert False
    return
