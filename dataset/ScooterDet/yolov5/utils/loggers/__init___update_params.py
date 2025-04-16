def update_params(self, params):
    if self.wandb:
        wandb.run.config.update(params, allow_val_change=True)
