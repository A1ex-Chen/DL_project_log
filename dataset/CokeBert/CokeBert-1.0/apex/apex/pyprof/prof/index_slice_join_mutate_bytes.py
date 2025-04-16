def bytes(self):
    tensor = self.tshape
    mask = self.mshape
    t = self.type
    b = 2 * Utility.numElems(tensor) * Utility.typeToBytes(t)
    b += Utility.numElems(mask)
    return b
