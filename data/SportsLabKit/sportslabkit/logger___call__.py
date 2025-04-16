def __call__(self, record: Mapping[str, Any]) ->bool:
    """Filter log records based on logging level.

        Args:
            record (Dict): Loguru record

        Returns:
            bool: True if record is at or above the logging level
        """
    levelno = logger.level(self.level).no
    return record['level'].no >= levelno
