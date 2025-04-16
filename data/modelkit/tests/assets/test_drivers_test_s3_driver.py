@skip_unless('ENABLE_S3_TEST', 'True')
def test_s3_driver(s3_assetsmanager):
    _perform_driver_test(s3_assetsmanager.storage_provider.driver)
