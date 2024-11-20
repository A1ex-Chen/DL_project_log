@property
@lru_cache(maxsize=2)
def xywh(self):
    """Returns boxes in [x, y, width, height] format."""
    return ops.xyxy2xywh(self.xyxy)
