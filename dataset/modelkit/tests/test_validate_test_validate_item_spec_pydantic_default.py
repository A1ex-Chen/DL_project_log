def test_validate_item_spec_pydantic_default():
    service_settings = LibrarySettings()


    class ItemType(pydantic.BaseModel):
        x: int
        y: str = 'ok'


    class ReturnType(pydantic.BaseModel):
        result: int
        something_else: str = 'ok'


    class TypedModel(Model[ItemType, ReturnType]):

        def _predict(self, item, **kwargs):
            return {'result': item.x + len(item.y)}
    m = TypedModel(service_settings=service_settings)
    res = m({'x': 10, 'y': 'okokokokok'})
    assert res.result == 20
    assert res.something_else == 'ok'
    res = m({'x': 10})
    assert res.result == 12
    assert res.something_else == 'ok'
    with pytest.raises(ItemValidationException):
        m({})
