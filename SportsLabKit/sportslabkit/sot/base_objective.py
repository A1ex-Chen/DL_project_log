def objective(trial: optuna.Trial):
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
    for attribute, param_values in params.items():
        for param_name, param_value in param_values.items():
            if attribute == 'self':
                setattr(self, param_name, param_value)
            else:
                setattr(getattr(self, attribute), param_name, param_value)
    self.reset()
    tracklet = self.track(frames)
    predictions = tracklet.get_observations('box')
    ground_truth_targets = [gt[0] for gt in ground_truth_positions]
    score = iou_scores(predictions, ground_truth_targets, xywh=True)
    return score
