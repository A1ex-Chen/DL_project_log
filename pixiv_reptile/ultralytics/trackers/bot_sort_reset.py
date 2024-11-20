def reset(self):
    """Reset tracker."""
    super().reset()
    self.gmc.reset_params()
