def __contains__(self, key):
    try:
        wr = WeakIdRef(key)
    except TypeError:
        return False
    return wr in self.data
