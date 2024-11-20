def __init__(self, half=False):
    """Initialize YOLOv8 ToTensor object with optional half-precision support."""
    super().__init__()
    self.half = half
