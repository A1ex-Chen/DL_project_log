def default_hp_search_backend():
    if is_optuna_available():
        return 'optuna'
    elif is_ray_available():
        return 'ray'
