def update(self, boxes=None, masks=None, probs=None, obb=None):
    """Updates detection results attributes including boxes, masks, probs, and obb with new data."""
    if boxes is not None:
        self.boxes = Boxes(ops.clip_boxes(boxes, self.orig_shape), self.
            orig_shape)
    if masks is not None:
        self.masks = Masks(masks, self.orig_shape)
    if probs is not None:
        self.probs = probs
    if obb is not None:
        self.obb = OBB(obb, self.orig_shape)
