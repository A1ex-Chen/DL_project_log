def bytes(self):
    return Utility.numElems(self.shape) * Utility.typeToBytes(self.type)
