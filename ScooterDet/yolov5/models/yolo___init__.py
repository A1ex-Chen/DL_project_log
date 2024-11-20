def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):
    super().__init__()
    self._from_detection_model(model, nc, cutoff
        ) if model is not None else self._from_yaml(cfg)
