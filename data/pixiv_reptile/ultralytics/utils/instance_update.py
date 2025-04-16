def update(self, bboxes, segments=None, keypoints=None):
    """Updates instance variables."""
    self._bboxes = Bboxes(bboxes, format=self._bboxes.format)
    if segments is not None:
        self.segments = segments
    if keypoints is not None:
        self.keypoints = keypoints
