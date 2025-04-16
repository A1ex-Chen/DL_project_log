def __init__(self, *keys, writer=None):
    self.writer = writer
    self._data = pd.DataFrame(index=keys, columns=['total', 'counts',
        'average'])
    self.reset()
