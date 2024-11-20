def test_local_driver(local_assetsmanager):
    local_driver = local_assetsmanager.storage_provider.driver
    _perform_driver_error_object_not_found(local_driver)
