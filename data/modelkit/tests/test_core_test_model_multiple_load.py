def test_model_multiple_load():
    loaded = 0


    class SomeModel(Model):
        CONFIGURATIONS = {'a': {}}

        def _load(self):
            nonlocal loaded
            loaded += 1

        def _predict(self, item):
            return self.some_attribute


    class SomeModel2(Model):
        CONFIGURATIONS = {'b': {'model_dependencies': {'a'}}}

        def _load(self):
            self.some_attribute = 'OK'

        def _predict(self, item):
            return self.some_attribute
    lib = ModelLibrary(models=[SomeModel, SomeModel2])
    lib.get('b')
    lib.get('a')
    assert loaded == 1
