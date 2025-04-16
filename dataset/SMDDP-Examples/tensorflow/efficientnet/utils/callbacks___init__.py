def __init__(self, batch_size, logger, logdir=None):
    """Callback for logging performance.

    Args:
      batch_size: Total batch size.
      log_steps: Interval of steps between logging of batch level stats.
      logdir: Optional directory to write TensorBoard summaries.
    """
    self.batch_size = batch_size
    self.global_steps = 0
    self.batch_time = []
    self.eval_time = 0
    super(EvalTimeHistory, self).__init__()
    self.logger = logger
