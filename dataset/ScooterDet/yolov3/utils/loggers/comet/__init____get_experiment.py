def _get_experiment(self, mode, experiment_id=None):
    if mode == 'offline':
        return comet_ml.ExistingOfflineExperiment(previous_experiment=
            experiment_id, **self.default_experiment_kwargs
            ) if experiment_id is not None else comet_ml.OfflineExperiment(**
            self.default_experiment_kwargs)
    try:
        if experiment_id is not None:
            return comet_ml.ExistingExperiment(previous_experiment=
                experiment_id, **self.default_experiment_kwargs)
        return comet_ml.Experiment(**self.default_experiment_kwargs)
    except ValueError:
        logger.warning(
            'COMET WARNING: Comet credentials have not been set. Comet will default to offline logging. Please set your credentials to enable online logging.'
            )
        return self._get_experiment('offline', experiment_id)
    return
