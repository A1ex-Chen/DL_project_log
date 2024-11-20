@rank_zero_only
def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
    save_json(pl_module.metrics, pl_module.metrics_save_path)
    return self._write_logs(trainer, pl_module, 'test')
