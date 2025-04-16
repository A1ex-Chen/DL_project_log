def denormalize(self, w, h):
    """Denormalizes boxes, segments, and keypoints from normalized coordinates."""
    if not self.normalized:
        return
    self._bboxes.mul(scale=(w, h, w, h))
    self.segments[..., 0] *= w
    self.segments[..., 1] *= h
    if self.keypoints is not None:
        self.keypoints[..., 0] *= w
        self.keypoints[..., 1] *= h
    self.normalized = False
