def bytes(self):
    b = self.elems() * Utility.typeToBytes(self.type)
    b *= 3 if self.dir == 'fprop' else 5
    return b
