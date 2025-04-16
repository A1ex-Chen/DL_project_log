def test_asset_dependencies():
    """
    Test that MyAsset can have model dependencies of any kind
    (SomeAsset, SomeModel, SomeAsyncModel), loaded and available during its _load.
    """


    class SomeAsset(Asset):
        CONFIGURATIONS = {'some_asset': {}}

        def _load(self) ->None:
            self.data = 'some_asset'


    class SomeModel(Model):
        CONFIGURATIONS = {'some_model': {}}

        def _load(self) ->None:
            self.data = 'some_model'

        def _predict(self, _):
            return self.data


    class SomeAsyncModel(AsyncModel):
        CONFIGURATIONS = {'some_async_model': {}}

        def _load(self) ->None:
            self.data = 'some_async_model'

        async def _predict(self, _):
            await asyncio.sleep(0)
            return self.data


    class MyAsset(Asset):
        CONFIGURATIONS = {'my_asset': {'model_dependencies': {'some_asset',
            'some_model', 'some_async_model'}}}

        def _load(self) ->None:
            self.data = {'my_asset': [self.model_dependencies['some_asset']
                .data, self.model_dependencies['some_model'].predict(None),
                self.model_dependencies['some_async_model'].predict(None)]}


    class MyModel(Model):
        CONFIGURATIONS = {'my_model': {'model_dependencies': {'my_asset'}}}

        def _predict(self, _):
            return self.model_dependencies['my_asset'].data
    model = load_model('my_model', models=[MyModel, MyAsset, SomeAsset,
        SomeModel, SomeAsyncModel])
    assert model.predict(None) == {'my_asset': ['some_asset', 'some_model',
        'some_async_model']}
