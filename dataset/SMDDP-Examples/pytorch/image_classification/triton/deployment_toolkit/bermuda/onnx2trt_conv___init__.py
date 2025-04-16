def __init__(self, *, max_batch_size: int, max_workspace_size: int,
    precision: str):
    self._max_batch_size = max_batch_size
    self._max_workspace_size = max_workspace_size
    self._precision = Precision(precision)
