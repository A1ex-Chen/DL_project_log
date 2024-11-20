def add(self, key, value):
    if self._cache_by_use.size >= self._max_size:
        removed = self._cache_by_use.remove_back()
        del self._cache_by_key[removed.key]
    node = self._cache_by_use.add_to_front(key, value)
    self._cache_by_key[key] = node
