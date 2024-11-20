def enumerate(self):
    return enumerate(zip(self.kernel, self.stride, self.num_repeat, self.
        expansion, self.channels))
