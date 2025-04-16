@pytest.fixture(scope='module')
def api_no_type(event_loop):
    np = pytest.importorskip('numpy')


    class ValidationNotSupported(Model[np.ndarray, np.ndarray]):
        CONFIGURATIONS = {'no_supported_model': {}}

        def _predict(self, item):
            return item
    router = ModelkitAutoAPIRouter(required_models=['unvalidated_model',
        'no_supported_model', 'some_model', 'some_complex_model',
        'some_asset', 'async_model'], models=[ValidationNotSupported,
        NotValidatedModel, SomeSimpleValidatedModel,
        SomeComplexValidatedModel, SomeAsset, SomeAsyncModel])
    app = fastapi.FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        yield client
