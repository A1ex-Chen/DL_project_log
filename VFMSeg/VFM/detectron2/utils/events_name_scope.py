@contextmanager
def name_scope(self, name):
    """
        Yields:
            A context within which all the events added to this storage
            will be prefixed by the name scope.
        """
    old_prefix = self._current_prefix
    self._current_prefix = name.rstrip('/') + '/'
    yield
    self._current_prefix = old_prefix
