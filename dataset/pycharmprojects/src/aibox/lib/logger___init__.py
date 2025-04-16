def __init__(self, logger: logging.Logger, enabled: bool):
    super().__init__()
    self._logger = logger
    self._enabled = enabled
