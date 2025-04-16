def flipud(self, h):
    """Flips the coordinates of bounding boxes, segments, and keypoints vertically."""
    if self._bboxes.format == 'xyxy':
        y1 = self.bboxes[:, 1].copy()
        y2 = self.bboxes[:, 3].copy()
        self.bboxes[:, 1] = h - y2
        self.bboxes[:, 3] = h - y1
    else:
        self.bboxes[:, 1] = h - self.bboxes[:, 1]
    self.segments[..., 1] = h - self.segments[..., 1]
    if self.keypoints is not None:
        self.keypoints[..., 1] = h - self.keypoints[..., 1]
