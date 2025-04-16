def get(self, model_key: str, item: Any, kwargs: Dict[str, Any]):
    cache_key = self.hash_key(model_key, item, kwargs)
    r = self.cache.get(cache_key)
    if r is None:
        return CacheItem(item, cache_key, None, True)
    return CacheItem(item, cache_key, r, False)
