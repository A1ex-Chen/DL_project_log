def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False,
    scaleup=True, center=True, stride=32):
    """Initialize LetterBox object with specific parameters."""
    self.new_shape = new_shape
    self.auto = auto
    self.scaleFill = scaleFill
    self.scaleup = scaleup
    self.stride = stride
    self.center = center
