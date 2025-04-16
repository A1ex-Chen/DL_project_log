def test_validate_none():
    service_settings = LibrarySettings()


    class SomeValidatedModel(Model):

        def _predict(self, item):
            return item
    m = SomeValidatedModel(service_settings=service_settings)
    assert m({'x': 10}) == {'x': 10}
    assert m(1) == 1
