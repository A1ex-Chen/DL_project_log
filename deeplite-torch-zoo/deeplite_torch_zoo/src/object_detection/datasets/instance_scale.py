def scale(self, scale_w, scale_h, bbox_only=False):
    """this might be similar with denormalize func but without normalized sign."""
    self._bboxes.mul(scale=(scale_w, scale_h, scale_w, scale_h))
    if bbox_only:
        return
    self.segments[..., 0] *= scale_w
    self.segments[..., 1] *= scale_h
    if self.keypoints is not None:
        self.keypoints[..., 0] *= scale_w
        self.keypoints[..., 1] *= scale_h
