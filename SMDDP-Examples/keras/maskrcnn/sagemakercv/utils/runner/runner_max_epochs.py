@property
def max_epochs(self):
    """
        Maximum training epochs.
        """
    return self.max_iters // self.steps_per_epoch + 1
