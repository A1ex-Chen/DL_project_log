def test_predict_override():


    class NoPredictModel(Model):
        pass
    with pytest.raises(NoPredictOverridenError):
        NoPredictModel()


    class BatchPredictModel(Model):

        def _predict_batch(self, items):
            return items
    m = BatchPredictModel()
    assert m._predict_mode == PredictMode.BATCH


    class SinglePredictModel(Model):

        def _predict(self, item):
            return item
    m = SinglePredictModel()
    assert m._predict_mode == PredictMode.SINGLE


    class BothPredictModel(Model):

        def _predict(self, item):
            return item

        def _predict_batch(self, items):
            return items
    m = SinglePredictModel()
    with pytest.raises(BothPredictsOverridenError):
        BothPredictModel()
