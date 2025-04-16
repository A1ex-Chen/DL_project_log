@skip_unless('ENABLE_AZ_TEST', 'True')
def test_az_driver_pickable(az_assetsmanager, monkeypatch):
    _perform_pickability_test(az_assetsmanager.storage_provider.driver,
        monkeypatch)
