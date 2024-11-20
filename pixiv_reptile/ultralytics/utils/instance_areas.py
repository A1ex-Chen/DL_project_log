def areas(self):
    """Return box areas."""
    return (self.bboxes[:, 2] - self.bboxes[:, 0]) * (self.bboxes[:, 3] -
        self.bboxes[:, 1]) if self.format == 'xyxy' else self.bboxes[:, 3
        ] * self.bboxes[:, 2]
