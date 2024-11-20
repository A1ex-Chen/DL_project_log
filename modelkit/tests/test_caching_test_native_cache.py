@pytest.mark.parametrize('cache_implementation', ['LFU', 'LRU', 'RR'])
def test_native_cache(cache_implementation):


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
        SomeModelValidated], settings={'cache': {'cache_provider': 'native',
        'implementation': cache_implementation, 'maxsize': 16}})
    assert isinstance(lib.cache, NativeCache)
    m = lib.get('model')
    m_multi = lib.get('model_multiple')
    ITEMS = [{'ok': {'boomer': 1}}, {'ok': {'boomer': [2, 2, 3]}}]
    _do_model_test(m, ITEMS)
    _do_model_test(m_multi, ITEMS)
    m_validated = lib.get('model_validated')
    _do_model_test(m_validated, ITEMS)
