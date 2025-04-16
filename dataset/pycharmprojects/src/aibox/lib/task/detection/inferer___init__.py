def __init__(self, model: Model, device_ids: List[int]=None):
    super().__init__()
    self._model = BunchDataParallel(model, device_ids)
