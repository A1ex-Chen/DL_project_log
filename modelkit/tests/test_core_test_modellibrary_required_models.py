def test_modellibrary_required_models():


    class SomeModel(Model):
        CONFIGURATIONS = {'yolo': {}, 'les simpsons': {}}

        def _predict(self, item):
            return item
    p = ModelLibrary(models=SomeModel)
    m = p.get('yolo')
    assert m
    assert m.configuration_key == 'yolo'
    assert m.__class__.__name__ == 'SomeModel'
    assert m.model_settings == {}
    assert m.asset_path == ''
    assert m.batch_size is None


    class SomeOtherModel(Model):
        pass
    with pytest.raises(ValueError):
        p.get('yolo', model_type=SomeOtherModel)
