def exception(self, message: str):
    if self._enabled:
        self._logger.exception(message)
