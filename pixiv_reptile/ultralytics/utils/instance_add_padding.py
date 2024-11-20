def add_padding(self, padw, padh):
    """Handle rect and mosaic situation."""
    assert not self.normalized, 'you should add padding with absolute coordinates.'
    self._bboxes.add(offset=(padw, padh, padw, padh))
    self.segments[..., 0] += padw
    self.segments[..., 1] += padh
    if self.keypoints is not None:
        self.keypoints[..., 0] += padw
        self.keypoints[..., 1] += padh
