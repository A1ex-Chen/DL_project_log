@property
def bbox_areas(self):
    """Calculate the area of bounding boxes."""
    return self._bboxes.areas()
