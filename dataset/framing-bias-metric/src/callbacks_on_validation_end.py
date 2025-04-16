@rank_zero_only
def on_validation_end(self, trainer: pl.Trainer, pl_module):
    save_json(pl_module.metrics, pl_module.metrics_save_path)
