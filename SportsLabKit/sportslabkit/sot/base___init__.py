def __init__(self, target, window_size=1, step_size=None, pre_init_args={},
    post_init_args={}):
    self.target = target
    self.init_target = target
    self.window_size = window_size
    self.step_size = step_size or window_size
    self.pre_init_args = pre_init_args
    self.post_init_args = post_init_args
    self.reset()
