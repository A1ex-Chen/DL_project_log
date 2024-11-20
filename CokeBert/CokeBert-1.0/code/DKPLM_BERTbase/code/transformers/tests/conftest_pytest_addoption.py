def pytest_addoption(parser):
    parser.addoption('--runslow', action='store_true', default=False, help=
        'run slow tests')
    parser.addoption('--use_cuda', action='store_true', default=False, help
        ='run tests on gpu')
