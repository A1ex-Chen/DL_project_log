def test_local_driver_overwrite(working_dir):
    driver = LocalStorageDriver(settings={'bucket': working_dir})
    driver.upload_object(os.path.join(TEST_DIR, 'assets', 'testdata',
        'some_data.json'), 'a/b/c')
    assert os.path.isfile(os.path.join(working_dir, 'a', 'b', 'c'))
    driver.upload_object(os.path.join(TEST_DIR, 'assets', 'testdata',
        'some_data.json'), 'a/b/c')
    assert os.path.isfile(os.path.join(working_dir, 'a', 'b', 'c'))
    driver.upload_object(os.path.join(TEST_DIR, 'assets', 'testdata',
        'some_data.json'), 'a/b')
    assert os.path.isfile(os.path.join(working_dir, 'a', 'b'))
    driver.upload_object(os.path.join(TEST_DIR, 'assets', 'testdata',
        'some_data.json'), 'a/b/c')
    assert os.path.isfile(os.path.join(working_dir, 'a', 'b', 'c'))
