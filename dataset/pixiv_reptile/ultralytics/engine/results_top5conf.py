@property
@lru_cache(maxsize=1)
def top5conf(self):
    """Returns confidence scores for the top 5 classification predictions."""
    return self.data[self.top5]
