def __init__(self):
    self._camera = None
    self._default_preprocessor = lambda x: x
    self.preprocessor = self._default_preprocessor
