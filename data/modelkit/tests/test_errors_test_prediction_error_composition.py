def test_prediction_error_composition(monkeypatch):
    monkeypatch.setenv('MODELKIT_ENABLE_SIMPLE_TRACEBACK', True)
    mm = OKModel(model_dependencies={'error_model': ErrorModel()})
    mm.load()
    with pytest.raises(CustomError) as excinfo:
        mm.predict({})
    assert len(excinfo.traceback) <= 4
    with pytest.raises(CustomError) as excinfo:
        mm.predict_batch([{}])
    assert len(excinfo.traceback) <= 4
    with pytest.raises(CustomError) as excinfo:
        next(mm.predict_gen(iter(({},))))
    assert len(excinfo.traceback) <= 4
