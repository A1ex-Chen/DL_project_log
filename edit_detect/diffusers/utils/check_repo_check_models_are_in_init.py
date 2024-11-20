def check_models_are_in_init():
    """Checks all models defined in the library are in the main init."""
    models_not_in_init = []
    dir_transformers = dir(diffusers)
    for module in get_model_modules():
        models_not_in_init += [model[0] for model in get_models(module,
            include_pretrained=True) if model[0] not in dir_transformers]
    models_not_in_init = [model for model in models_not_in_init if not
        is_a_private_model(model)]
    if len(models_not_in_init) > 0:
        raise Exception(
            f"The following models should be in the main init: {','.join(models_not_in_init)}."
            )
