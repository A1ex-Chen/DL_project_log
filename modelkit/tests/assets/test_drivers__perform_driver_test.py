def _perform_driver_test(driver):
    assert not driver.exists('some/object')
    with tempfile.TemporaryDirectory() as tempd:
        with open(os.path.join(tempd, 'name'), 'w') as fsrc:
            fsrc.write('some contents')
        driver.upload_object(os.path.join(tempd, 'name'), 'some/object')
    assert driver.exists('some/object')
    with tempfile.TemporaryDirectory() as tempdir:
        temp_path = os.path.join(tempdir, 'test')
        driver.download_object('some/object', temp_path)
        with open(temp_path) as fdst:
            assert fdst.read() == 'some contents'
    assert [x for x in driver.iterate_objects()] == ['some/object']
    driver.delete_object('some/object')
    assert not driver.exists('some/object')
