def bytes(self):
    b = self.elems() * (Utility.typeToBytes(self.stype) + Utility.
        typeToBytes(self.dtype))
    return b
