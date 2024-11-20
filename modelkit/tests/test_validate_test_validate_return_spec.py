def test_validate_return_spec():
    service_settings = LibrarySettings()


    class ItemModel(pydantic.BaseModel):
        x: int


    class SomeValidatedModel(Model[Any, ItemModel]):

        def _predict(self, item):
            return item
    m = SomeValidatedModel(service_settings=service_settings)
    ret = m({'x': 10})
    assert ret.x == 10
    with pytest.raises(ReturnValueValidationException):
        m({'x': 'something', 'blabli': 10})
