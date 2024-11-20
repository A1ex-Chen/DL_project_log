@skip_unless('ENABLE_GCS_TEST', 'True')
def test_gcs_driver(gcs_assetsmanager):
    gcs_driver = gcs_assetsmanager.storage_provider.driver
    _perform_driver_error_object_not_found(gcs_driver)
