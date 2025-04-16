def mul(self, scale):
    """
        Args:
            scale (tuple | list | int): the scale for four coords.
        """
    if isinstance(scale, Number):
        scale = to_4tuple(scale)
    assert isinstance(scale, (tuple, list))
    assert len(scale) == 4
    self.bboxes[:, 0] *= scale[0]
    self.bboxes[:, 1] *= scale[1]
    self.bboxes[:, 2] *= scale[2]
    self.bboxes[:, 3] *= scale[3]
