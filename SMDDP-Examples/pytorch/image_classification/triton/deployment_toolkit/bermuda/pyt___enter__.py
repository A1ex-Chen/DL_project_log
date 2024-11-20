def __enter__(self):
    self._output_names = list(self._model.outputs)
    return self
