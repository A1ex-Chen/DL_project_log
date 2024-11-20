def on_fit_epoch_end(trainer):
    """Log training metrics at the end of each fit epoch to MLflow."""
    if mlflow:
        mlflow.log_metrics(metrics=sanitize_dict(trainer.metrics), step=
            trainer.epoch)
