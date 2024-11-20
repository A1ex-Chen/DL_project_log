@skip_unless('ENABLE_GCS_TEST', 'True')
def test_gcs_driver_pickable(gcs_assetsmanager, monkeypatch):
    _perform_pickability_test(gcs_assetsmanager.storage_provider.driver,
        monkeypatch)
