@property
@lru_cache(maxsize=1)
def xy(self):
    """Returns x, y coordinates of keypoints."""
    return self.data[..., :2]
