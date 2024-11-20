def test_cache_empty(self):
    cache = lru.LRUCache()
    self.assertIsNone(cache.query(123))
    self.assertEqual(len(cache._cache_by_key), 0)
    self.assertEqual(cache._cache_by_use.size, 0)
