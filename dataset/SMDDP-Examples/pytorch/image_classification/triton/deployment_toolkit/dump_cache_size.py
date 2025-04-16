@property
def cache_size(self):
    return {name: sum([a.nbytes for a in data.values()]) for name, data in
        self._items_cache.items()}
