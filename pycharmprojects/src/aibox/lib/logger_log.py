def log(self, level: int, message: str):
    if self._enabled:
        self._logger.log(level, message)
