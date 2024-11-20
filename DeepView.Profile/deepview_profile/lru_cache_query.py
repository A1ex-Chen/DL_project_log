def query(self, key):
    if key not in self._cache_by_key:
        return None
    node = self._cache_by_key[key]
    self._cache_by_use.move_to_front(node)
    return node.value
