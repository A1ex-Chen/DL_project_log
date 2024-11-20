def check_model_list():
    """Check the model list inside the transformers library."""
    models_dir = os.path.join(PATH_TO_DIFFUSERS, 'models')
    _models = []
    for model in os.listdir(models_dir):
        model_dir = os.path.join(models_dir, model)
        if os.path.isdir(model_dir) and '__init__.py' in os.listdir(model_dir):
            _models.append(model)
    models = [model for model in dir(diffusers.models) if not model.
        startswith('__')]
    missing_models = sorted(set(_models).difference(models))
    if missing_models:
        raise Exception(
            f"The following models should be included in {models_dir}/__init__.py: {','.join(missing_models)}."
            )
