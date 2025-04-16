@property
@lru_cache(maxsize=1)
def top5(self):
    """Return the indices of the top 5 class probabilities."""
    return (-self.data).argsort(0)[:5].tolist()
