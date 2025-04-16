@skip_unless('ENABLE_AZ_TEST', 'True')
def test_az_driver(az_assetsmanager):
    _perform_driver_test(az_assetsmanager.storage_provider.driver)
