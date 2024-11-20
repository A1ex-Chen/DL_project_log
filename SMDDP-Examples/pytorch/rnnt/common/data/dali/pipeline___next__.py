def __next__(self):
    if self.i == self.n:
        raise StopIteration
    ret = self.pert_coeff[self.i]
    self.i += 1
    return ret
