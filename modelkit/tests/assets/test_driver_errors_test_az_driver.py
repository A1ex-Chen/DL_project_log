@skip_unless('ENABLE_AZ_TEST', 'True')
def test_az_driver(az_assetsmanager):
    az_driver = az_assetsmanager.storage_provider.driver
    _perform_driver_error_object_not_found(az_driver)
