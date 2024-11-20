def create_modelkit_app(models=None, required_models=None):
    """
    Creates a modelkit FastAPI app with the specified models and required models.

    This is meant to be used in conjunction with gunicorn or uvicorn in order to
     start a server.

    Run with:
    ```
    export MODELKIT_REQUIRED_MODELS=... # optional
    export MODELKIT_DEFAULT_PACKAGE=... # mandatory
    gunicorn --workers 4             --preload             --worker-class=uvicorn.workers.UvicornWorker             'modelkit.api.create_modelkit_app()'
    ```
    """
    if not (models or os.environ.get('MODELKIT_DEFAULT_PACKAGE')):
        raise ModelsNotFound(
            'Please add `your_package` as argument or set the `MODELKIT_DEFAULT_PACKAGE=your_package` env variable.'
            )
    if os.environ.get('MODELKIT_REQUIRED_MODELS') and not required_models:
        required_models = os.environ.get('MODELKIT_REQUIRED_MODELS').split(':')
    app = fastapi.FastAPI()
    router = ModelkitAutoAPIRouter(required_models=required_models, models=
        models)
    app.include_router(router)
    return app
