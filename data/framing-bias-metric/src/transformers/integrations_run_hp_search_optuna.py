def run_hp_search_optuna(trainer, n_trials: int, direction: str, **kwargs
    ) ->BestRun:

    def _objective(trial, checkpoint_dir=None):
        model_path = None
        if checkpoint_dir:
            for subdir in os.listdir(checkpoint_dir):
                if subdir.startswith(PREFIX_CHECKPOINT_DIR):
                    model_path = os.path.join(checkpoint_dir, subdir)
        trainer.objective = None
        trainer.train(model_path=model_path, trial=trial)
        if getattr(trainer, 'objective', None) is None:
            metrics = trainer.evaluate()
            trainer.objective = trainer.compute_objective(metrics)
        return trainer.objective
    timeout = kwargs.pop('timeout', None)
    n_jobs = kwargs.pop('n_jobs', 1)
    study = optuna.create_study(direction=direction, **kwargs)
    study.optimize(_objective, n_trials=n_trials, timeout=timeout, n_jobs=
        n_jobs)
    best_trial = study.best_trial
    return BestRun(str(best_trial.number), best_trial.value, best_trial.params)
