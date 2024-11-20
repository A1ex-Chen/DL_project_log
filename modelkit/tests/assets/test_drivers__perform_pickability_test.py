def _perform_pickability_test(driver, monkeypatch):
    monkeypatch.setattr(driver, '_client', None)
    assert pickle.dumps(driver)
