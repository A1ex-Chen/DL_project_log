@pytest.fixture(scope='function')
def base_dir():
    with tempfile.TemporaryDirectory() as base_dir:
        yield base_dir
