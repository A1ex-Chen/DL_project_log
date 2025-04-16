def run_hp_search_ray(trainer, n_trials: int, direction: str, **kwargs
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
            trainer._tune_save_checkpoint()
            ray.tune.report(objective=trainer.objective, **metrics, done=True)
    _tb_writer = trainer.pop_callback(TensorBoardCallback)
    trainer.model = None
    if 'resources_per_trial' not in kwargs and trainer.args.n_gpu > 0:
        n_jobs = int(kwargs.pop('n_jobs', 1))
        num_gpus_per_trial = trainer.args.n_gpu
        if num_gpus_per_trial / n_jobs >= 1:
            num_gpus_per_trial = int(math.ceil(num_gpus_per_trial / n_jobs))
        kwargs['resources_per_trial'] = {'gpu': num_gpus_per_trial}
    if 'progress_reporter' not in kwargs:
        from ray.tune import CLIReporter
        kwargs['progress_reporter'] = CLIReporter(metric_columns=['objective'])
    if 'keep_checkpoints_num' in kwargs and kwargs['keep_checkpoints_num'] > 0:
        trainer.use_tune_checkpoints = True
        if kwargs['keep_checkpoints_num'] > 1:
            logger.warning(
                'Currently keeping {} checkpoints for each trial. Checkpoints are usually huge, consider setting `keep_checkpoints_num=1`.'
                )
    if 'scheduler' in kwargs:
        from ray.tune.schedulers import ASHAScheduler, HyperBandForBOHB, MedianStoppingRule, PopulationBasedTraining
        if isinstance(kwargs['scheduler'], PopulationBasedTraining):
            if not trainer.use_tune_checkpoints:
                logger.warning(
                    "You are using PopulationBasedTraining but you haven't enabled checkpointing. This means your trials will train from scratch everytime they are exploiting new configurations. Consider enabling checkpointing by passing `keep_checkpoints_num=1` as an additional argument to `Trainer.hyperparameter_search`."
                    )
        if isinstance(kwargs['scheduler'], (ASHAScheduler,
            MedianStoppingRule, HyperBandForBOHB, PopulationBasedTraining)
            ) and (not trainer.args.do_eval or trainer.args.
            evaluation_strategy == EvaluationStrategy.NO):
            raise RuntimeError(
                "You are using {cls} as a scheduler but you haven't enabled evaluation during training. This means your trials will not report intermediate results to Ray Tune, and can thus not be stopped early or used to exploit other trials parameters. If this is what you want, do not use {cls}. If you would like to use {cls}, make sure you pass `do_eval=True` and `evaluation_strategy='steps'` in the Trainer `args`."
                .format(cls=type(kwargs['scheduler']).__name__))
    analysis = ray.tune.run(_objective, config=trainer.hp_space(None),
        num_samples=n_trials, **kwargs)
    best_trial = analysis.get_best_trial(metric='objective', mode=direction[:3]
        )
    best_run = BestRun(best_trial.trial_id, best_trial.last_result[
        'objective'], best_trial.config)
    if _tb_writer is not None:
        trainer.add_callback(_tb_writer)
    return best_run
