def test_required_models():


    class SomeModel(Model):
        CONFIGURATIONS = {'model': {}}

        def _predict(self, item):
            return item


    class SomeOtherModel(Model):
        CONFIGURATIONS = {'other_model': {}}

        def _predict(self, item):
            return item
    lib = ModelLibrary(required_models=[], models=[SomeModel, SomeOtherModel])
    assert len(lib.models) == 0
    assert lib.required_models == {}
    lib = ModelLibrary(models=[SomeModel, SomeOtherModel])
    assert len(lib.models) == 2
    assert lib.required_models == {'model': {}, 'other_model': {}}
