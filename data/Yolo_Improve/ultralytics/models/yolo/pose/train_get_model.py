def get_model(self, cfg=None, weights=None, verbose=True):
    """Get pose estimation model with specified configuration and weights."""
    model = PoseModel(cfg, ch=3, nc=self.data['nc'], data_kpt_shape=self.
        data['kpt_shape'], verbose=verbose)
    if weights:
        model.load(weights)
    return model
