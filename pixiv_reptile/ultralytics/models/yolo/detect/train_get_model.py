def get_model(self, cfg=None, weights=None, verbose=True):
    """Return a YOLO detection model."""
    model = DetectionModel(cfg, nc=self.data['nc'], verbose=verbose and 
        RANK == -1)
    if weights:
        model.load(weights)
    return model
