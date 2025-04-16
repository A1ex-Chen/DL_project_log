def __exit__(self, exc_type, exc_value, traceback):
    self._input_names = None
    self._output_names = None
    self._session = None
    self._recover_env_variables(self._old_env_values)
