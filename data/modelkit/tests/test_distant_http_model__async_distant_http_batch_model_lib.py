def _async_distant_http_batch_model_lib(event_loop, **settings):


    class TestModel(AsyncDistantHTTPBatchModel):
        CONFIGURATIONS = {'test_async_distant_http_batch_model': {
            'model_settings': {'endpoint':
            'http://127.0.0.1:8000/api/path/endpoint/batch', 'async_mode': 
            True, **settings}}}
    lib = ModelLibrary(models=[TestModel])
    yield lib
    event_loop.run_until_complete(lib.aclose())
