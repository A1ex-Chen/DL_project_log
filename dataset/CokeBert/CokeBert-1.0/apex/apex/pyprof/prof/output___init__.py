def __init__(self, args):
    self.cols = args.c
    self.csv = args.csv
    self.col = True if args.w > 0 else False
    self.width = args.w
    w = 0
    for col in self.cols:
        assert col in Output.table.keys()
        w += Output.table[col][3]
    if self.col and w > self.width:
        print('Minimum width required to print {} = {}. Exiting.'.format(
            ','.join(self.cols), w))
        sys.exit(1)
    remainder = self.width - w
    if 'kernel' in self.cols and 'params' in self.cols:
        Output.table['kernel'][3] = int(remainder / 2)
        Output.table['params'][3] = int(remainder / 2)
    elif 'kernel' in self.cols:
        Output.table['kernel'][3] = remainder
    elif 'params' in self.cols:
        Output.table['params'][3] = remainder
    cadena = ''
    for col in self.cols:
        _, _, t, w = Output.table[col]
        cadena += '%-{}.{}s '.format(w, w)
    self.hFormat = cadena
    cadena = ''
    for col in self.cols:
        _, _, t, w = Output.table[col]
        if t == str:
            cadena += '%-{}.{}s '.format(w, w)
        elif t == int:
            cadena += '%{}d '.format(w)
    self.dFormat = cadena
