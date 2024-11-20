def test_load_model():


    class SomeModel(Model):
        CONFIGURATIONS = {'model': {}}

        def _predict(self, item):
            return item
    m = load_model('model', models=SomeModel)
    assert m({'ok': 'boomer'}) == {'ok': 'boomer'}
