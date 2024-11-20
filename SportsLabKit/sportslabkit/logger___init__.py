def __init__(self, level: str='INFO'):
    """Filter log records based on logging level.

        Args:
            level (str, optional): Logging level to filter on. Defaults to "INFO".
        """
    self._level = level
