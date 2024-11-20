@lru_cache(maxsize=None)
def warn_once(message):
    """
    Write a warning message only once.
    """
    warn(message)
