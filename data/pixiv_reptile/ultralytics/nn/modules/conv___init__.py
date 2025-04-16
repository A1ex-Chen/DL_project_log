def __init__(self, dimension=1):
    """Concatenates a list of tensors along a specified dimension."""
    super().__init__()
    self.d = dimension
