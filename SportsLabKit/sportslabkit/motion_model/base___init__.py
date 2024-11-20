def __init__(self, is_multi_target=False):
    """Initialize the MotionModel."""
    self.input_is_batched = False
    self.name = self.__class__.__name__
    self.is_multi_target = is_multi_target
