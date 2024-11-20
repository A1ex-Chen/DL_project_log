def __init__(self, use_scale=False, **kwargs):
    super(scaled_attention, self).__init__(**kwargs)
    self.use_scale = use_scale
