def test_internal_error(monkeypatch):
    m = OKModel()


    class CustomError(BaseException):
        pass

    def _buggy_predict_cache_items(*args, **kwargs):
        raise CustomError
        yield None
    monkeypatch.setattr(m, '_predict_cache_items', _buggy_predict_cache_items)
    with pytest.raises(CustomError):
        m({})
    with pytest.raises(CustomError):
        m.predict({})
    with pytest.raises(CustomError):
        m.predict_batch([{}])
    with pytest.raises(CustomError):
        for _ in m.predict_gen([{}]):
            pass
