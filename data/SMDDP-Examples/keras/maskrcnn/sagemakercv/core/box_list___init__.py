def __init__(self, boxes):
    """Constructs box collection.

    Args:
      boxes: a tensor of shape [N, 4] representing box corners

    Raises:
      ValueError: if invalid dimensions for bbox data or if bbox data is not in
          float32 format.
    """
    if len(boxes.get_shape()) != 2 or boxes.get_shape()[-1] != 4:
        raise ValueError('Invalid dimensions for box data.')
    if boxes.dtype != tf.float32:
        raise ValueError('Invalid tensor type: should be tf.float32')
    self.data = {'boxes': boxes}
