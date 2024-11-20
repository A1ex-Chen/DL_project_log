@skip_unless('ENABLE_S3_TEST', 'True')
def test_s3_driver_pickable(s3_assetsmanager, monkeypatch):
    _perform_pickability_test(s3_assetsmanager.storage_provider.driver,
        monkeypatch)
