def hp_params(trial):
    if is_optuna_available():
        if isinstance(trial, optuna.Trial):
            return trial.params
    if is_ray_available():
        if isinstance(trial, dict):
            return trial
    raise RuntimeError(f'Unknown type for trial {trial.__class__}')
