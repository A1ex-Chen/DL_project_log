def test_cache_provider_settings(monkeypatch):
    monkeypatch.setenv('MODELKIT_CACHE_PROVIDER', 'redis')
    lib_settings = LibrarySettings()
    assert isinstance(lib_settings.cache, RedisSettings)
    assert lib_settings.cache.cache_provider == 'redis'
    monkeypatch.setenv('MODELKIT_CACHE_PROVIDER', 'native')
    lib_settings = LibrarySettings()
    assert isinstance(lib_settings.cache, NativeCacheSettings)
    assert lib_settings.cache.cache_provider == 'native'
    monkeypatch.setenv('MODELKIT_CACHE_PROVIDER', 'none')
    assert LibrarySettings().cache is None
    monkeypatch.setenv('MODELKIT_CACHE_PROVIDER', 'not supported')
    assert LibrarySettings().cache is None
    monkeypatch.delenv('MODELKIT_CACHE_PROVIDER')
    assert LibrarySettings().cache is None
