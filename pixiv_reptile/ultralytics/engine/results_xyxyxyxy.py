@property
@lru_cache(maxsize=2)
def xyxyxyxy(self):
    """Convert OBB format to 8-point (xyxyxyxy) coordinate format of shape (N, 4, 2) for rotated bounding boxes."""
    return ops.xywhr2xyxyxyxy(self.xywhr)
