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
