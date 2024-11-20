def _check_model_names(models):
    model_names = []
    for model in models:
        if not hasattr(model, 'name'):
            raise ValueError('models must have name attr')
        model_names.append(model.name)
    if len(model_names) != len(set(model_names)):
        raise ValueError('models must have unique name: {}'.format(', '.
            join(model_names)))
