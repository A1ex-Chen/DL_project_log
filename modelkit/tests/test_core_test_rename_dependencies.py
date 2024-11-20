def test_rename_dependencies():


    class SomeModel(Model):
        CONFIGURATIONS = {'ok': {}}

        def _predict(self, item):
            return self.configuration_key


    class SomeModel2(Model):
        CONFIGURATIONS = {'boomer': {}}

        def _predict(self, item):
            return self.configuration_key


    class FinalModel(Model):
        CONFIGURATIONS = {'model_no_rename': {'model_dependencies': {'ok'}},
            'model_rename': {'model_dependencies': {'ok': 'boomer'}}}

        def _predict(self, item):
            return self.model_dependencies['ok'](item)
    lib = ModelLibrary(models=[SomeModel, SomeModel2, FinalModel])
    assert lib.get('model_no_rename')({}) == 'ok'
    assert lib.get('model_rename')({}) == 'boomer'
