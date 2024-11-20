def remove_zero_area_boxes(self):
    """Remove zero-area boxes, i.e. after clipping some boxes may have zero width or height."""
    good = self.bbox_areas > 0
    if not all(good):
        self._bboxes = self._bboxes[good]
        if len(self.segments):
            self.segments = self.segments[good]
        if self.keypoints is not None:
            self.keypoints = self.keypoints[good]
    return good
