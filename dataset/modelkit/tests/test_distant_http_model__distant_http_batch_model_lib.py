def _distant_http_batch_model_lib(**settings):


    class TestModel(DistantHTTPBatchModel):
        CONFIGURATIONS = {'test_distant_http_batch_model': {
            'model_settings': {'endpoint':
            'http://127.0.0.1:8000/api/path/endpoint/batch', 'async_mode': 
            False, **settings}}}
    lib = ModelLibrary(models=[TestModel])
    yield lib
    lib.close()
