def num_layers(self):
    _f = lambda l: len(set(map(len, l)))
    l = [self.kernel, self.stride, self.num_repeat, self.expansion, self.
        channels]
    assert _f(l) == 1
    return len(self.kernel)
