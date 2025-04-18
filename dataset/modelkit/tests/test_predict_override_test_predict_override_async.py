def test_predict_override_async():


    class NoPredictModel(AsyncModel):
        pass
    with pytest.raises(NoPredictOverridenError):
        NoPredictModel()


    class BatchPredictModel(AsyncModel):

        async def _predict_batch(self, items):
            return items
    m = BatchPredictModel()
    assert m._predict_mode == PredictMode.BATCH


    class SinglePredictModel(AsyncModel):

        async def _predict(self, item):
            return item
    m = SinglePredictModel()
    assert m._predict_mode == PredictMode.SINGLE


    class BothPredictModel(AsyncModel):

        async def _predict(self, item):
            return item

        async def _predict_batch(self, items):
            return items
    m = SinglePredictModel()
    with pytest.raises(BothPredictsOverridenError):
        BothPredictModel()
