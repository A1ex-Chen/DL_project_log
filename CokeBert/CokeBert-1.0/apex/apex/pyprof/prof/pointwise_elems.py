def elems(self):
    tensor = self.shape
    t = self.type
    if len(tensor) == 1:
        elems = 2 * Utility.numElems(tensor[0])
    elif len(tensor) == 2:
        if tensor[0] == tensor[1]:
            elems = Utility.numElems(tensor[0])
            if self.dir == 'fprop':
                elems *= 3
            elif self.op_ in ['add', '__add__', 'sub', '__sub__', '__isub__']:
                elems *= 2
            elif self.op_ in ['__mul__', '__rmul__', 'div', '__truediv__']:
                elems *= 3
            else:
                assert False
        else:
            array1 = np.empty(list(tensor[0]))
            array2 = np.empty(list(tensor[1]))
            try:
                out = np.broadcast(array1, array2).shape
            except:
                assert False
            elems = Utility.numElems(tensor[0])
            elems += Utility.numElems(tensor[1])
            elems += Utility.numElems(out)
    elif len(tensor) == 3:
        if tensor[0] == tensor[1] == tensor[2]:
            elems = Utility.numElems(tensor[0])
            elems *= 4
        else:
            assert False
    else:
        assert False
    return elems
