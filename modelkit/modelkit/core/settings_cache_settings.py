def cache_settings():
    s = CacheSettings()
    if s.cache_provider == 'none':
        return None
    elif s.cache_provider == 'redis':
        return RedisSettings()
    elif s.cache_provider == 'native':
        return NativeCacheSettings()
    else:
        return None
