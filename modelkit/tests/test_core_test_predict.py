def test_predict():


    class SomeModel(Model):

        def _predict(self, item):
            return item
    m = SomeModel()
    assert m.predict({}) == {}


    class SomeModelBatch(Model):

        def _predict_batch(self, items):
            return items
    m = SomeModelBatch()
    assert m.predict({}) == {}
