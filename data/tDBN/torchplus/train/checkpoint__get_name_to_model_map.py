def _get_name_to_model_map(models):
    if isinstance(models, dict):
        name_to_model = {name: m for name, m in models.items()}
    else:
        _check_model_names(models)
        name_to_model = {m.name: m for m in models}
    return name_to_model
