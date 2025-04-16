def test_compose_sync_async():


    class SomeAsyncModel(AsyncModel):
        CONFIGURATIONS = {'async_model': {}}

        async def _predict(self, item, **kwargs):
            await asyncio.sleep(0)
            return item


    class ComposedModel(Model):
        CONFIGURATIONS = {'composed_model': {'model_dependencies': {
            'async_model'}}}

        def _predict(self, item, **kwargs):
            self.model_dependencies['async_model'].predict_batch([item])
            return self.model_dependencies['async_model'].predict(item)
    library = ModelLibrary(models=[SomeAsyncModel, ComposedModel])
    m = library.get('composed_model')
    assert isinstance(m.model_dependencies['async_model'], WrappedAsyncModel)
    assert m.predict({'hello': 'world'}) == {'hello': 'world'}
