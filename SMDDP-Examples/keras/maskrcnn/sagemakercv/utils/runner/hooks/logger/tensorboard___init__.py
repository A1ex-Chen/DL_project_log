def __init__(self, name='metrics', re_match='.*loss', interval=100,
    image_interval=None, ignore_last=True, reset_flag=True):
    super(TensorboardMetricsLogger, self).__init__(interval, ignore_last,
        reset_flag)
    self.name = name
    if isinstance(re_match, str):
        self.re_match = [re_match]
    else:
        self.re_match = re_match
