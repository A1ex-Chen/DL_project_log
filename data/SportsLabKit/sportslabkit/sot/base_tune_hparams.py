def tune_hparams(self, frames, ground_truth_positions, n_trials=100,
    hparam_search_space=None, metric=iou_scores, verbose=False,
    return_study=False):

    def objective(trial: optuna.Trial):
        params = {}
        for attribute, param_space in hparams.items():
            params[attribute] = {}
            for param_name, param_values in param_space.items():
                if param_values['type'] == 'categorical':
                    params[attribute][param_name] = trial.suggest_categorical(
                        param_name, param_values['values'])
                elif param_values['type'] == 'float':
                    params[attribute][param_name] = trial.suggest_float(
                        param_name, param_values['low'], param_values['high'])
                elif param_values['type'] == 'logfloat':
                    params[attribute][param_name] = trial.suggest_float(
                        param_name, param_values['low'], param_values[
                        'high'], log=True)
                elif param_values['type'] == 'int':
                    params[attribute][param_name] = trial.suggest_int(
                        param_name, param_values['low'], param_values['high'])
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
    if isinstance(ground_truth_positions, BBoxDataFrame):
        ground_truth_positions = np.expand_dims(ground_truth_positions.
            values, axis=1)[:, :, :4]
    hparams = self.create_hparam_dict()
    print('Hyperparameter search space: ')
    for attribute, param_space in hparams.items():
        print(f'{attribute}:')
        for param_name, param_values in param_space.items():
            print(f'\t{param_name}: {param_values}')
    if verbose:
        optuna.logging.set_verbosity(optuna.logging.INFO)
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    best_iou = study.best_value
    if return_study:
        return best_params, best_iou, study
    return best_params, best_iou
