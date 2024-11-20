def normalize(self, w, h):
    """Normalize bounding boxes, segments, and keypoints to image dimensions."""
    if self.normalized:
        return
    self._bboxes.mul(scale=(1 / w, 1 / h, 1 / w, 1 / h))
    self.segments[..., 0] /= w
    self.segments[..., 1] /= h
    if self.keypoints is not None:
        self.keypoints[..., 0] /= w
        self.keypoints[..., 1] /= h
    self.normalized = True
