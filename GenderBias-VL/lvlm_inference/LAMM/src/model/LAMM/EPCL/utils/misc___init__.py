def __init__(self, window_size=20, fmt=None):
    if fmt is None:
        fmt = '{median:.4f} ({global_avg:.4f})'
    self.deque = deque(maxlen=window_size)
    self.total = 0.0
    self.count = 0
    self.fmt = fmt
