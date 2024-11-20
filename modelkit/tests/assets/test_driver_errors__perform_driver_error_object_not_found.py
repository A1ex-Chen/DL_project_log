def _perform_driver_error_object_not_found(driver):
    with pytest.raises(errors.ObjectDoesNotExistError):
        driver.download_object('someasset', 'somedestination')
    assert not os.path.isfile('somedestination')
