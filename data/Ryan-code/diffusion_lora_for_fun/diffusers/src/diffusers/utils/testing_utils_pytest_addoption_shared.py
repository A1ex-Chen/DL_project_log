def pytest_addoption_shared(parser):
    """
    This function is to be called from `conftest.py` via `pytest_addoption` wrapper that has to be defined there.

    It allows loading both `conftest.py` files at once without causing a failure due to adding the same `pytest`
    option.

    """
    option = '--make-reports'
    if option not in pytest_opt_registered:
        parser.addoption(option, action='store', default=False, help=
            'generate report files. The value of this option is used as a prefix to report names'
            )
        pytest_opt_registered[option] = 1
