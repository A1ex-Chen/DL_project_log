def add(self, offset):
    """
        Args:
            offset (tuple | list | int): the offset for four coords.
        """
    if isinstance(offset, Number):
        offset = to_4tuple(offset)
    assert isinstance(offset, (tuple, list))
    assert len(offset) == 4
    self.bboxes[:, 0] += offset[0]
    self.bboxes[:, 1] += offset[1]
    self.bboxes[:, 2] += offset[2]
    self.bboxes[:, 3] += offset[3]
