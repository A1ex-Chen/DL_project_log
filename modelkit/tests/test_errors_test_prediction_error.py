@pytest.mark.parametrize('model', [ErrorModel(), ErrorBatchModel()])
def test_prediction_error(monkeypatch, model):
    monkeypatch.setenv('MODELKIT_ENABLE_SIMPLE_TRACEBACK', True)
    with pytest.raises(CustomError) as excinfo:
        model.predict({})
    assert len(excinfo.traceback) <= 3
    with pytest.raises(CustomError) as excinfo:
        model.predict_batch([{}])
    assert len(excinfo.traceback) <= 3
    with pytest.raises(CustomError) as excinfo:
        next(model.predict_gen(iter(({},))))
    assert len(excinfo.traceback) <= 3
