@property
def average_steps_per_second(self):
    """The average training steps per second across all epochs."""
    return (self.global_steps - 1) / self.eval_time
