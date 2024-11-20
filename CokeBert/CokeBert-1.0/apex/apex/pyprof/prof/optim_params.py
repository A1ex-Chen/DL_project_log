def params(self):
    p = OrderedDict([('T', self.w['shape']), ('wtype', self.w['dtype']), (
        'gtype', self.g['dtype'])])
    return p
