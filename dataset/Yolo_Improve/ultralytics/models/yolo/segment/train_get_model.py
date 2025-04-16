def get_model(self, cfg=None, weights=None, verbose=True):
    """Return SegmentationModel initialized with specified config and weights."""
    model = SegmentationModel(cfg, ch=3, nc=self.data['nc'], verbose=
        verbose and RANK == -1)
    if weights:
        model.load(weights)
    return model
