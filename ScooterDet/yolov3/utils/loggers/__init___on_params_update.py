def on_params_update(self, params: dict):
    if self.wandb:
        self.wandb.wandb_run.config.update(params, allow_val_change=True)
    if self.comet_logger:
        self.comet_logger.on_params_update(params)
