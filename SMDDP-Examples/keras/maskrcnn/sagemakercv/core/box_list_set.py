def set(self, boxes):
    """Convenience function for setting box coordinates.

    Args:
      boxes: a tensor of shape [N, 4] representing box corners

    Raises:
      ValueError: if invalid dimensions for bbox data
    """
    if len(boxes.get_shape()) != 2 or boxes.get_shape()[-1] != 4:
        raise ValueError('Invalid dimensions for box data.')
    self.data['boxes'] = boxes
