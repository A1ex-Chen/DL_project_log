def __init__(self, model, cfg, device=None, **kwargs):
    super().__init__()
    self._device = device
    self.depth_range = cfg['depth_range']
    self.n_max_network_queries = cfg['n_max_network_queries']
    self.white_background = cfg['white_background']
    self.cfg = cfg
    self.model = model.to(device)
