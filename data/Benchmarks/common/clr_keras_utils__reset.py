def _reset(self, new_base_lr=None, new_max_lr=None, new_step_size=None):
    """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
    if new_base_lr is not None:
        self.base_lr = new_base_lr
    if new_max_lr is not None:
        self.max_lr = new_max_lr
    if new_step_size is not None:
        self.step_size = new_step_size
    self.clr_iterations = 0.0
