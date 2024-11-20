def decode_bboxes(self, bboxes, anchors):
    """Decode rotated bounding boxes."""
    return dist2rbox(bboxes, self.angle, anchors, dim=1)
