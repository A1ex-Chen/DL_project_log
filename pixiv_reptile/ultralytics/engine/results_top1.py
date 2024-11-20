@property
@lru_cache(maxsize=1)
def top1(self):
    """Return the index of the class with the highest probability."""
    return int(self.data.argmax())
