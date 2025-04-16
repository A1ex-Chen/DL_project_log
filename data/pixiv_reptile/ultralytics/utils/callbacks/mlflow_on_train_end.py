def on_train_end(trainer):
    """Log model artifacts at the end of the training."""
    if not mlflow:
        return
    mlflow.log_artifact(str(trainer.best.parent))
    for f in trainer.save_dir.glob('*'):
        if f.suffix in {'.png', '.jpg', '.csv', '.pt', '.yaml'}:
            mlflow.log_artifact(str(f))
    keep_run_active = os.environ.get('MLFLOW_KEEP_RUN_ACTIVE', 'False').lower(
        ) == 'true'
    if keep_run_active:
        LOGGER.info(
            f'{PREFIX}mlflow run still alive, remember to close it using mlflow.end_run()'
            )
    else:
        mlflow.end_run()
        LOGGER.debug(f'{PREFIX}mlflow run ended')
    LOGGER.info(
        f"""{PREFIX}results logged to {mlflow.get_tracking_uri()}
{PREFIX}disable with 'yolo settings mlflow=False'"""
        )
