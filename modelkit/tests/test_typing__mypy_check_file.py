def _mypy_check_file(fn, raises):
    result = mypy.api.run([os.path.join(TEST_DIR, 'testdata', 'typing', fn)])
    if raises:
        assert result[2] != 0, result
    else:
        assert result[2] == 0, result
