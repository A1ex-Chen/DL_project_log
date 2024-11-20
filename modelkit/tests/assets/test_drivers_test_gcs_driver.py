@skip_unless('ENABLE_GCS_TEST', 'True')
def test_gcs_driver(gcs_assetsmanager):
    _perform_driver_test(gcs_assetsmanager.storage_provider.driver)
