@property
@lru_cache(maxsize=1)
def top1conf(self):
    """Retrieves the confidence score of the highest probability class."""
    return self.data[self.top1]
