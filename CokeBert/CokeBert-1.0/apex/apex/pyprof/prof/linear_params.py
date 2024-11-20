def params(self):
    m, n, k, x, w, t = self.m, self.n, self.k, self.x, self.w, self.type
    if len(n) == 1:
        n = n[0]
    if self.op_ == 'linear':
        if self.dir == 'fprop':
            p = OrderedDict([('M', m), ('N', n), ('K', k), ('type', t)])
        elif self.dir == 'bprop':
            if self.sub == 0:
                p = OrderedDict([('M', k), ('N', n), ('K', m), ('type', t)])
            elif self.sub == 1:
                p = OrderedDict([('M', k), ('N', m), ('K', n), ('type', t)])
            else:
                p = OrderedDict([('X', x), ('W', w), ('type', t)])
        else:
            assert False
    elif self.op_ == 'bias':
        p = OrderedDict([('M', m), ('N', n), ('type', t)])
    else:
        assert False
    return p
