@property
@lru_cache(maxsize=2)
def xywhn(self):
    """Returns normalized bounding boxes in [x, y, width, height] format."""
    xywh = ops.xyxy2xywh(self.xyxy)
    xywh[..., [0, 2]] /= self.orig_shape[1]
    xywh[..., [1, 3]] /= self.orig_shape[0]
    return xywh
