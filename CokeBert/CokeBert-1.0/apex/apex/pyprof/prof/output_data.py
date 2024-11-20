def data(self, a):
    if a.dir == '':
        direc = 'na'
    else:
        direc = a.dir
    if a.op == '':
        op = 'na'
    else:
        op = a.op
    if a.mod == '':
        mod = 'na'
    else:
        mod = a.mod
    cadena = ()
    for col in self.cols:
        attr = Output.table[col][1]
        val = getattr(a, attr)
        if col == 'layer':
            assert type(val) == list
            val = ':'.join(val)
            val = '-' if val == '' else val
        if col == 'trace':
            assert type(val) == list
            if self.col and len(val):
                val = val[-1]
                val = val.split('/')[-1]
            else:
                val = ','.join(val)
                val = '-' if val == '' else val
        if col in ['seq', 'altseq']:
            assert type(val) == list
            val = ','.join(map(str, val))
            val = '-' if val == '' else val
        cadena = cadena + (val,)
    self.foo(cadena, self.dFormat)
