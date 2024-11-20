def bytes(self):
    e = self.elems()
    if self.dir == 'fprop':
        e *= 4
    else:
        e *= 5
    return e * Utility.typeToBytes(self.type)
