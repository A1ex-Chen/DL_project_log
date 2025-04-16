def set_verbosity(self, verbosity_level):
    """Sets the threshold for what messages will be logged."""
    if self._logger is not None:
        self._logger.setLevel(verbosity_level)
        for handler in self._logger.handlers:
            handler.setLevel(verbosity_level)
