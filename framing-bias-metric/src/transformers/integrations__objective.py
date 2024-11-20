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
