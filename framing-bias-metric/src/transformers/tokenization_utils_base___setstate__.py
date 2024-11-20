def __setstate__(self, state):
    if 'data' in state:
        self.data = state['data']
    if 'encodings' in state:
        self._encodings = state['encodings']
