def pytest_collection_modifyitems(config, items):
    if config.getoption('--runslow'):
        return
    skip_slow = pytest.mark.skip(reason='need --runslow option to run')
    for item in items:
        if 'slow' in item.keywords:
            item.add_marker(skip_slow)
