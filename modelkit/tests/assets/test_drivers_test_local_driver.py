def test_local_driver(local_assetsmanager):
    _perform_driver_test(local_assetsmanager.storage_provider.driver)
