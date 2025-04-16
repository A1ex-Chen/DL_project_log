def __init__(self, implementation, maxsize):
    self.cache: cachetools.Cache = self.NATIVE_CACHE_IMPLEMENTATIONS[
        implementation](maxsize)
