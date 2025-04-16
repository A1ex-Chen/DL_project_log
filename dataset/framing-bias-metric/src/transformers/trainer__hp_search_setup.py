def _hp_search_setup(self, trial: Union['optuna.Trial', Dict[str, Any]]):
    """ HP search setup code """
    self._trial = trial
    if self.hp_search_backend is None or trial is None:
        return
    params = self.hp_space(trial
        ) if self.hp_search_backend == HPSearchBackend.OPTUNA else trial
    for key, value in params.items():
        if not hasattr(self.args, key):
            raise AttributeError(
                f'Trying to set {key} in the hyperparameter search but there is no corresponding field in `TrainingArguments`.'
                )
        old_attr = getattr(self.args, key, None)
        if old_attr is not None:
            value = type(old_attr)(value)
        setattr(self.args, key, value)
    if self.hp_search_backend == HPSearchBackend.OPTUNA:
        logger.info('Trial:', trial.params)
