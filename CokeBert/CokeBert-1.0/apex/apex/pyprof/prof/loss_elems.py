def elems(self):
    red = self.red
    e = Utility.numElems(self.shape)
    if self.dir == 'fprop':
        if red == 'none':
            e *= 3
        else:
            e *= 2
    elif red == 'none':
        e *= 4
    else:
        e *= 3
    return e
