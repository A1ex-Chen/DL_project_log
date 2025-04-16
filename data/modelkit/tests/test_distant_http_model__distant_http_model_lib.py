def _distant_http_model_lib(**settings):


    class TestModel(DistantHTTPModel):
        CONFIGURATIONS = {'test_distant_http_model': {'model_settings': {
            'endpoint': 'http://127.0.0.1:8000/api/path/endpoint',
            'async_mode': False, **settings}}}
    lib = ModelLibrary(models=[TestModel])
    yield lib
    lib.close()
