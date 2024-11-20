def __len__(self):
    """Return the number of detections in the Results object from a non-empty attribute set (boxes, masks, etc.)."""
    for k in self._keys:
        v = getattr(self, k)
        if v is not None:
            return len(v)
