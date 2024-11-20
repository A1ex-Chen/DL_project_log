def on_params_update(self, params):
    if self.wandb:
        self.wandb.wandb_run.config.update(params, allow_val_change=True)
