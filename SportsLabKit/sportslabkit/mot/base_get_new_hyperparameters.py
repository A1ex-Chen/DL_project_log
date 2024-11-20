def get_new_hyperparameters(self, hparams, trial):
    params = {}
    for attribute, param_space in hparams.items():
        params[attribute] = {}
        for param_name, param_values in param_space.items():
            if param_values['type'] == 'categorical':
                params[attribute][param_name] = trial.suggest_categorical(
                    param_name, param_values['values'])
            elif param_values['type'] == 'float':
                params[attribute][param_name] = trial.suggest_float(param_name,
                    param_values['low'], param_values['high'])
            elif param_values['type'] == 'logfloat':
                params[attribute][param_name] = trial.suggest_float(param_name,
                    param_values['low'], param_values['high'], log=True)
            elif param_values['type'] == 'int':
                params[attribute][param_name] = trial.suggest_int(param_name,
                    param_values['low'], param_values['high'])
            else:
                raise ValueError(
                    f"Unknown parameter type: {param_values['type']}")
    return params
