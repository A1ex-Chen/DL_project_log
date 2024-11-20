def _log_model(experiment, trainer):
    """Log the best-trained model to Comet.ml."""
    model_name = _get_comet_model_name()
    experiment.log_model(model_name, file_or_folder=str(trainer.best),
        file_name='best.pt', overwrite=True)
