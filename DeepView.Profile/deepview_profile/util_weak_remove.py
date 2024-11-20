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
