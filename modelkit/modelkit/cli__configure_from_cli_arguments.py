def _configure_from_cli_arguments(models, required_models, settings):
    models = list(models) or None
    required_models = list(required_models) or None
    if not (models or os.environ.get('MODELKIT_DEFAULT_PACKAGE')):
        raise ModelsNotFound(
            'Please add `your_package` as argument or set the `MODELKIT_DEFAULT_PACKAGE=your_package` env variable.'
            )
    service = ModelLibrary(models=models, required_models=required_models,
        settings=settings)
    return service
