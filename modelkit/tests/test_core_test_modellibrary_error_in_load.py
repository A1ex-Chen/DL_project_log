@pytest.mark.parametrize('error', [KeyError, ZeroDivisionError])
def test_modellibrary_error_in_load(error):


    class SomeModel(Model):
        CONFIGURATIONS = {'model': {}}

        def _load(self):
            raise error

        def _predict(self, item):
            return item
    library = ModelLibrary(models=SomeModel, settings={'lazy_loading': True})
    try:
        library.get('model')
        assert False
    except error as err:
        assert 'not loaded' not in str(err)
