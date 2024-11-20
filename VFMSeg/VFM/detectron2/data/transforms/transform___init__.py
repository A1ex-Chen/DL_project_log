def __init__(self, op):
    """
        Args:
            op (Callable): operation to be applied to the image,
                which takes in a PIL Image and returns a transformed
                PIL Image.
                For reference on possible operations see:
                - https://pillow.readthedocs.io/en/stable/
        """
    if not callable(op):
        raise ValueError('op parameter should be callable')
    super().__init__(op)
