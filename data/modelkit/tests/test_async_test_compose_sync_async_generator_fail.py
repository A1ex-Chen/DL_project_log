def test_compose_sync_async_generator_fail():


    class SomeAsyncModel(AsyncModel):
        CONFIGURATIONS = {'async_model': {}}

        async def _predict(self, item, **kwargs):
            await asyncio.sleep(0)
            return item

        async def close(self):
            await asyncio.sleep(0)


    class ComposedModel(Model):
        CONFIGURATIONS = {'composed_model': {'model_dependencies': {
            'async_model'}}}

        def _predict(self, item, **kwargs):
            for r in AsyncToSync(self.model_dependencies['async_model'].
                async_model.predict_gen)(iter((item,))):
                break
            return r
    library = ModelLibrary(models=[SomeAsyncModel, ComposedModel])
    m = library.get('composed_model')
    assert isinstance(m.model_dependencies['async_model'], WrappedAsyncModel)
    with pytest.raises(TypeError):
        assert m.predict({'hello': 'world'}) == {'hello': 'world'}
    library.close()
