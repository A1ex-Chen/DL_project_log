@skip_unless('ENABLE_REDIS_TEST', 'True')
def test_redis_cache(redis_service):


    class SomeModel(Model):
        CONFIGURATIONS = {'model': {'model_settings': {'cache_predictions':
            True}}}

        def _predict(self, item):
            return item


    class SomeModelMultiple(Model):
        CONFIGURATIONS = {'model_multiple': {'model_settings': {
            'cache_predictions': True}}}

        def _predict_batch(self, items):
            return items


    class Item(pydantic.BaseModel):


        class SubItem(pydantic.BaseModel):
            boomer: Union[int, List[int]]
        ok: SubItem


    class SomeModelValidated(Model[Item, Item]):
        CONFIGURATIONS = {'model_validated': {'model_settings': {
            'cache_predictions': True}}}

        def _predict_batch(self, items):
            return items
    lib = ModelLibrary(models=[SomeModel, SomeModelMultiple,
        SomeModelValidated], settings={'cache': {'cache_provider': 'redis'}})
    assert isinstance(lib.cache, RedisCache)
    m = lib.get('model')
    m_multi = lib.get('model_multiple')
    ITEMS = [{'ok': {'boomer': 1}}, {'ok': {'boomer': [2, 2, 3]}}]
    _do_model_test(m, ITEMS)
    _do_model_test(m_multi, ITEMS)
    m_validated = lib.get('model_validated')
    _do_model_test(m_validated, ITEMS)
