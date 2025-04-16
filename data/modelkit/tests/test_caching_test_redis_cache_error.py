def test_redis_cache_error(monkeypatch):


    class SomeModelValidated(Model):
        CONFIGURATIONS = {'model': {'model_settings': {'cache_predictions':
            True}}}

        def _predict_batch(self, items):
            return items
    with pytest.raises(modelkit.utils.redis.RedisCacheException):
        ModelLibrary(models=[SomeModelValidated], settings={'cache': {
            'cache_provider': 'redis'}})
