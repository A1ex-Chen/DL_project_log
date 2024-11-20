def on_pretrain_routine_start(trainer):
    """Creates or resumes a CometML experiment at the start of a YOLO pre-training routine."""
    experiment = comet_ml.get_global_experiment()
    is_alive = getattr(experiment, 'alive', False)
    if not experiment or not is_alive:
        _create_experiment(trainer.args)
