@pytest.fixture(scope='module')
def async_distant_http_batch_model_lib(event_loop):
    yield from _async_distant_http_batch_model_lib(event_loop)
