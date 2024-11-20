def __init__(self, init_scale=2 ** 32, scale_factor=2.0, scale_window=1000):
    self.cur_scale = init_scale
    self.cur_iter = 0
    self.last_overflow_iter = -1
    self.scale_factor = scale_factor
    self.scale_window = scale_window
