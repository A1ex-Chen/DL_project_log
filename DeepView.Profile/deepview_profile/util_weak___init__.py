def __init__(self, dict=None):
    self.data = {}

    def remove(k, selfref=ref(self)):
        self = selfref()
        if self is not None:
            if self._iterating:
                self._pending_removals.append(k)
            else:
                try:
                    del self.data[k]
                except KeyError:
                    pass
    self._remove = remove
    self._pending_removals = []
    self._iterating = set()
    self._dirty_len = False
    if dict is not None:
        self.update(dict)
