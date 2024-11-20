@skip_unless('ENABLE_S3_TEST', 'True')
def test_s3_driver(s3_assetsmanager):
    s3_driver = s3_assetsmanager.storage_provider.driver
    _perform_driver_error_object_not_found(s3_driver)
