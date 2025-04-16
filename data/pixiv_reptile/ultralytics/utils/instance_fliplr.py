def fliplr(self, w):
    """Reverses the order of the bounding boxes and segments horizontally."""
    if self._bboxes.format == 'xyxy':
        x1 = self.bboxes[:, 0].copy()
        x2 = self.bboxes[:, 2].copy()
        self.bboxes[:, 0] = w - x2
        self.bboxes[:, 2] = w - x1
    else:
        self.bboxes[:, 0] = w - self.bboxes[:, 0]
    self.segments[..., 0] = w - self.segments[..., 0]
    if self.keypoints is not None:
        self.keypoints[..., 0] = w - self.keypoints[..., 0]
