@property
def id(self):
    """Return the tracking IDs of the oriented bounding boxes (if available)."""
    return self.data[:, -3] if self.is_track else None
