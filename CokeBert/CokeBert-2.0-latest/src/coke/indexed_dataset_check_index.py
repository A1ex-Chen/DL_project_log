def check_index(self, i):
    if i < 0 or i >= self.size:
        raise IndexError('index out of range')
