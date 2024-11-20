def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
    rank_zero_info('***** Validation results *****')
    metrics = trainer.callback_metrics
    for key in sorted(metrics):
        if key not in ['log', 'progress_bar']:
            rank_zero_info('{} = {}\n'.format(key, str(metrics[key])))
