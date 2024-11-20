def flops(self):
    direction = self.dir
    tensor = self.i['shape']
    t = self.i['dtype']
    elems = Utility.numElems(tensor)
    return elems
