@staticmethod
def next_id():
    """Increment and return the global track ID counter."""
    BaseTrack._count += 1
    return BaseTrack._count
