def areas(self):
    """Return box areas."""
    self.convert('xyxy')
    return (self.bboxes[:, 2] - self.bboxes[:, 0]) * (self.bboxes[:, 3] -
        self.bboxes[:, 1])
