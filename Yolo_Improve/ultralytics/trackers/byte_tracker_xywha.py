@property
def xywha(self):
    """Get current position in bounding box format (center x, center y, width, height, angle)."""
    if self.angle is None:
        LOGGER.warning(
            'WARNING ⚠️ `angle` attr not found, returning `xywh` instead.')
        return self.xywh
    return np.concatenate([self.xywh, self.angle[None]])
