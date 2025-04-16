@contextmanager
def temp_verbosity(self, verbosity_level):
    """Sets the a temporary threshold for what messages will be logged."""
    if self._logger is not None:
        old_verbosity = self.get_verbosity()
        try:
            self.set_verbosity(verbosity_level)
            yield
        finally:
            self.set_verbosity(old_verbosity)
    else:
        try:
            yield
        finally:
            pass
