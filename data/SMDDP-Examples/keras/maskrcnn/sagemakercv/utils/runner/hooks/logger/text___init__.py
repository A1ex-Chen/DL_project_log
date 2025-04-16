def __init__(self, interval=25, ignore_last=True, reset_flag=False):
    super(TextLoggerHook, self).__init__(interval, ignore_last, reset_flag)
    self.time_sec_tot = 0
