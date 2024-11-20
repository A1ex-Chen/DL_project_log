def __init__(self, boxes, orig_shape) ->None:
    """Initialize an OBB instance with oriented bounding box data and original image shape."""
    if boxes.ndim == 1:
        boxes = boxes[None, :]
    n = boxes.shape[-1]
    assert n in {7, 8}, f'expected 7 or 8 values but got {n}'
    super().__init__(boxes, orig_shape)
    self.is_track = n == 8
    self.orig_shape = orig_shape
