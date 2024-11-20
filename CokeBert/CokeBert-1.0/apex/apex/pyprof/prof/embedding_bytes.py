def bytes(self):
    ishape = self.ishape
    itype = self.itype
    eshape = self.eshape
    etype = self.etype
    ielems = Utility.numElems(ishape)
    b = 0
    if self.dir == 'fprop':
        b += ielems * Utility.typeToBytes(itype)
        b += ielems * eshape[1] * 2 * Utility.typeToBytes(etype)
    else:
        b = ielems * eshape[1] * 3 * Utility.typeToBytes(etype)
        if self.sub > 0:
            b = 0
    return b
