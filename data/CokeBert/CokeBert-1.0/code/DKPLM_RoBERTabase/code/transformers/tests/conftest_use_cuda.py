@pytest.fixture
def use_cuda(request):
    """ Run test on gpu """
    return request.config.getoption('--use_cuda')
