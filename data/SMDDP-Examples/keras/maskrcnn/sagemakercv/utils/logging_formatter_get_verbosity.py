def get_verbosity(self):
    """Return how much logging output will be produced."""
    if self._logger is not None:
        return self._logger.getEffectiveLevel()
