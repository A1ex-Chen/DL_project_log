@pytest.fixture(scope='module')
def distant_http_batch_model_lib():
    yield from _distant_http_batch_model_lib()
