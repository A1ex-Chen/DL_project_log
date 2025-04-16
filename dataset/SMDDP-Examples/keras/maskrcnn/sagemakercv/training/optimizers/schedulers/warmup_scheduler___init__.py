def __init__(self, scheduler, warmup_ratio, warmup_steps, overlap=False,
    warmup_type='linear', dtype=tf.float32):
    super(LinearWarmup, self).__init__()
    self.scheduler = scheduler
    self.warmup_ratio = warmup_ratio
    self.warmup_steps = tf.cast(warmup_steps, dtype)
    self.warmup_type = warmup_type
    self.dtype = dtype
    self.scheduler_learning_rate = scheduler(0)
    self.initial_learning_rate = tf.cast(self.scheduler_learning_rate *
        self.warmup_ratio, dtype)
    self.overlap = overlap
