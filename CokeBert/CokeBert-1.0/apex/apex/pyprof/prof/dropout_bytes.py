def bytes(self):
    return Utility.typeToBytes(self.type) * self.elems() * 2
