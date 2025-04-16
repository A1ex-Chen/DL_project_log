@level.setter
def level(self, level: str) ->None:
    """Sets the logging level.

        Args:
            level (str): Logging level
        """
    level = level.upper()
    self._level = level
