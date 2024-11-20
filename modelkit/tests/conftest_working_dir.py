@pytest.fixture(scope='function')
def working_dir(base_dir):
    working_dir = os.path.join(base_dir, 'working_dir')
    os.makedirs(working_dir)
    yield working_dir
