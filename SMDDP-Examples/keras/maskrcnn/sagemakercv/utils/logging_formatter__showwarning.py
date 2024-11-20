def _showwarning(self, message, category, filename, lineno, file=None, line
    =None):
    """
        Implementation of showwarnings which redirects to logging.
        It will call warnings.formatwarning and will log the resulting string
        with level logging.WARNING.
        """
    s = warnings.formatwarning(message, category, filename, lineno, line)
    self.warning('%s', s)
