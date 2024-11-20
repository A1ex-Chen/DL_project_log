def add_to_front(self, key, value):
    node = _LRUCacheNode(key, value)
    self._add_to_front(node)
    self.size += 1
    return node
