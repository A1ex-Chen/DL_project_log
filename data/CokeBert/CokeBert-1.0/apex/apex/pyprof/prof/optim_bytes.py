def bytes(self):
    wshape = self.w['shape']
    wtype = self.w['dtype']
    gtype = self.g['dtype']
    b = 0
    elems = Utility.numElems(wshape)
    b += 6 * elems * Utility.typeToBytes(wtype)
    b += elems * Utility.typeToBytes(gtype)
    if wtype != gtype:
        b += elems * Utility.typeToBytes(gtype)
    return b
