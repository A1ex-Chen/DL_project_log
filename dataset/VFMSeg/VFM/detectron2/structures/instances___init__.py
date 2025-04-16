def __init__(self, image_size: Tuple[int, int], **kwargs: Any):
    """
        Args:
            image_size (height, width): the spatial size of the image.
            kwargs: fields to add to this `Instances`.
        """
    self._image_size = image_size
    self._fields: Dict[str, Any] = {}
    for k, v in kwargs.items():
        self.set(k, v)
