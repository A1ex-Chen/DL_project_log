def __init__(self, temperature=10000, normalize=False, scale=None, pos_type
    ='fourier', d_pos=None, d_in=3, gauss_scale=1.0):
    super().__init__()
    self.temperature = temperature
    self.normalize = normalize
    if scale is not None and normalize is False:
        raise ValueError('normalize should be True if scale is passed')
    if scale is None:
        scale = 2 * math.pi
    assert pos_type in ['sine', 'fourier']
    self.pos_type = pos_type
    self.scale = scale
    if pos_type == 'fourier':
        assert d_pos is not None
        assert d_pos % 2 == 0
        B = torch.empty((d_in, d_pos // 2)).normal_()
        B = B * gauss_scale
        self.register_buffer('gauss_B', B)
        self.d_pos = d_pos
