def history(self, name):
    """
        Returns:
            HistoryBuffer: the scalar history for name
        """
    ret = self._history.get(name, None)
    if ret is None:
        raise KeyError('No history metric available for {}!'.format(name))
    return ret
