def on_train_epoch_end(trainer):
    """Log training metrics at the end of each train epoch to MLflow."""
    if mlflow:
        mlflow.log_metrics(metrics={**sanitize_dict(trainer.lr), **
            sanitize_dict(trainer.label_loss_items(trainer.tloss, prefix=
            'train'))}, step=trainer.epoch)
