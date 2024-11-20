def __init__(self, num_pos_feats=64, temperature=10000, normalize=False,
    scale=None):
    super().__init__()
    self.num_pos_feats = num_pos_feats
    self.temperature = temperature
    self.normalize = normalize
    if scale is not None and normalize is False:
        raise ValueError('normalize should be True if scale is passed')
    if scale is None:
        scale = 2 * math.pi
    self.scale = scale
