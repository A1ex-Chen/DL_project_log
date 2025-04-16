def __exit__(self, exc_type, exc_value, traceback):
    self._output_names = None
    self._model = None
