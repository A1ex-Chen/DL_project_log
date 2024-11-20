def __init__(self, window_size=1, step_size=None, max_staleness=5,
    min_length=5, callbacks=None):
    self.window_size = window_size
    self.step_size = step_size or window_size
    self.max_staleness = max_staleness
    self.min_length = min_length
    self.trial_params = []
    self.callbacks = callbacks or []
    self.reset()
