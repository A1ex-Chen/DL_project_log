def params(self):
    p = OrderedDict([('T', self.shape), ('stype', self.stype), ('dtype',
        self.dtype)])
    return p
