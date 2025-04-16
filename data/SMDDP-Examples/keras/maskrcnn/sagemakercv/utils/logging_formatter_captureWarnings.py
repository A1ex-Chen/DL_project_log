def captureWarnings(self, capture):
    """
        If capture is true, redirect all warnings to the logging package.
        If capture is False, ensure that warnings are not redirected to logging
        but to their original destinations.
        """
    if self._logger is not None:
        if capture and self.old_warnings_showwarning is None:
            self.old_warnings_showwarning = warnings.showwarning
            warnings.showwarning = self._showwarning
        elif not capture and self.old_warnings_showwarning is not None:
            warnings.showwarning = self.old_warnings_showwarning
            self.old_warnings_showwarning = None
