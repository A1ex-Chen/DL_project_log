def test_local_driver_pickable(local_assetsmanager, monkeypatch):
    _perform_pickability_test(local_assetsmanager.storage_provider.driver,
        monkeypatch)
