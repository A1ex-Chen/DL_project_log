@property
def result(self):
    """Get current tracking results."""
    coords = self.xyxy if self.angle is None else self.xywha
    return coords.tolist() + [self.track_id, self.score, self.cls, self.idx]
