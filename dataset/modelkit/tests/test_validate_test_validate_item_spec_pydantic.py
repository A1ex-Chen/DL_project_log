def test_validate_item_spec_pydantic():
    service_settings = LibrarySettings()


    class ItemModel(pydantic.BaseModel):
        x: int


    class SomeValidatedModel(Model[ItemModel, Any]):

        def _predict(self, item):
            return item
    valid_test_item = {'x': 10}
    m = SomeValidatedModel(service_settings=service_settings)
    assert m(valid_test_item).model_dump() == valid_test_item
    with pytest.raises(ItemValidationException):
        m({'ok': 1})
    with pytest.raises(ItemValidationException):
        m({'x': 'something', 'blabli': 10})
    assert [x.model_dump() for x in m.predict_batch([valid_test_item] * 2)
        ] == [valid_test_item] * 2
