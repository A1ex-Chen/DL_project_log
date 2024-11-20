def __exit__(self, exc_type, exc_value, traceback):
    self._context.__exit__(exc_type, exc_value, traceback)
    self._input_names = None
    self._output_names = None
    self._buffers = None
