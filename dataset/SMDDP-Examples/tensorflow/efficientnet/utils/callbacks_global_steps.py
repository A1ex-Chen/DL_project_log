@property
def global_steps(self):
    """The current 1-indexed global step."""
    return self.steps_before_epoch + self.steps_in_epoch
