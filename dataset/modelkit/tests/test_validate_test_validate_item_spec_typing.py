def test_validate_item_spec_typing():
    service_settings = LibrarySettings()


    class SomeValidatedModel(Model[Dict[str, int], Any]):

        def _predict(self, item):
            return item
    valid_test_item = {'x': 10}
    m = SomeValidatedModel(service_settings=service_settings)
    assert m(valid_test_item) == valid_test_item
    with pytest.raises(ItemValidationException):
        m.predict_batch(['ok'])
    with pytest.raises(ItemValidationException):
        m('x')
    with pytest.raises(ItemValidationException):
        m.predict_batch([1, 2, 1])
    assert m.predict_batch([valid_test_item] * 2) == [valid_test_item] * 2
