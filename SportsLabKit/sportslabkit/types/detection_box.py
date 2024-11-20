@box.setter
def box(self, value: Box):
    value = np.array(value).squeeze()
    if value.shape != (4,):
        raise ValueError(
            f'box should have the shape (4, ), but got {value.shape}')
    self._box = value
