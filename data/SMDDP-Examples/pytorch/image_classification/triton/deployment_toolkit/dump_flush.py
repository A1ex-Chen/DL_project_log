def flush(self):
    for prefix, data in self._items_cache.items():
        self._dump(prefix, data)
    self._items_cache = {}
