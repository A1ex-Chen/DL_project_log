def __init__(self, cfg, renderer, depth_estimator=None, device=None, **kwargs):
    super().__init__()
    self.renderer = renderer.to(device)
    if depth_estimator is not None:
        self.depth_estimator = depth_estimator.to(device)
    else:
        self.depth_estimator = None
    self.device = device
