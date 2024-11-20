def reset_params(self) ->None:
    """Reset parameters."""
    self.prevFrame = None
    self.prevKeyPoints = None
    self.prevDescriptors = None
    self.initializedFirstFrame = False
