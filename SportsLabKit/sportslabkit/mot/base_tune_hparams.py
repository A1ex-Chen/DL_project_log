def tune_hparams(self, frames_list, bbdf_gt_list, n_trials=100,
    hparam_search_space=None, verbose=False, return_study=False, use_bbdf=
    False, reuse_detections=False, sampler=None, pruner=None):

    def objective(trial: optuna.Trial):
        params = self.get_new_hyperparameters(hparams, trial)
        self.trial_params.append(params)
        self.apply_hyperparameters(params)
        scores = []
        for i, (frames, bbdf_gt) in enumerate(zip(frames_list, bbdf_gt_list)):
            self.reset()
            if reuse_detections:
                self.detection_model = self.detection_models[i]
            try:
                bbdf_pred = self.track(frames)
            except ValueError as e:
                logger.error(e)
                return np.nan
            score = hota_score(bbdf_pred, bbdf_gt)['HOTA']
            scores.append(score)
            trial.report(np.mean(scores), step=len(scores))
            if trial.should_prune():
                raise optuna.TrialPruned()
        return np.mean(scores)
    hparams = hparam_search_space or self.create_hparam_dict()
    logger.info('Hyperparameter search space:')
    for attribute, param_space in hparams.items():
        logger.info(f'{attribute}:')
        for param_name, param_values in param_space.items():
            logger.info(f'\t{param_name}: {param_values}')
    if verbose:
        optuna.logging.set_verbosity(optuna.logging.INFO)
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    if use_bbdf:
        raise NotImplementedError
    if reuse_detections:
        self.detection_models = []
        for frames in frames_list:
            list_of_detections = []
            for frame in tqdm(frames, desc='Detecting frames for reuse'):
                list_of_detections.append(self.detection_model(frame)[0])
            dummy_detection_model = DummyDetectionModel(list_of_detections)
            og_detection_model = self.detection_model
            self.detection_models.append(dummy_detection_model)
    if sampler is None:
        sampler = optuna.samplers.TPESampler(multivariate=True)
    if pruner is None:
        pruner = optuna.pruners.MedianPruner()
    self.trial_params = []
    study = optuna.create_study(direction='maximize', sampler=sampler,
        pruner=pruner)
    study.optimize(objective, n_trials=n_trials)
    if reuse_detections:
        self.detection_model = og_detection_model
    best_value = study.best_value
    self.best_params = self.trial_params[study.best_trial.number]
    self.apply_hyperparameters(self.best_params)
    if return_study:
        return self.best_params, best_value, study
    return self.best_params, best_value
