def test_validate_list_items():
    service_settings = LibrarySettings()


    class ItemModel(pydantic.BaseModel):
        x: str
        y: str = 'ok'


    class SomeValidatedModel(Model[ItemModel, Any]):

        def __init__(self, *args, **kwargs):
            self.counter = 0
            super().__init__(*args, **kwargs)

        def _predict(self, item):
            self.counter += 1
            return item
    m = SomeValidatedModel(service_settings=service_settings)
    m.predict_batch([{'x': '10', 'y': 'ko'}] * 10)
    assert m.counter == 10
    m({'x': '10', 'y': 'ko'})
    assert m.counter == 11
