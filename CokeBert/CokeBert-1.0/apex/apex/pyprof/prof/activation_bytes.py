def bytes(self):
    direction = self.dir
    tensor = self.i['shape']
    t = self.i['dtype']
    elems = Utility.numElems(tensor)
    elems = elems * (2 if direction == 'fprop' else 3)
    return elems * Utility.typeToBytes(t)
