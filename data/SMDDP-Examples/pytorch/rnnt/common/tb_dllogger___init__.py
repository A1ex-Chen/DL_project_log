def __init__(self, enabled, log_dir, name, interval=1, dummies=True):
    self.enabled = enabled
    self.interval = interval
    self.cache = {}
    if self.enabled:
        self.summary_writer = SummaryWriter(log_dir=os.path.join(log_dir,
            name), flush_secs=120, max_queue=200)
        atexit.register(self.summary_writer.close)
        if dummies:
            for key in ('aaa', 'zzz'):
                self.summary_writer.add_scalar(key, 0.0, 1)
