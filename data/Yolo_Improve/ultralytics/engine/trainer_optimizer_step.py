def optimizer_step(self):
    """Perform a single step of the training optimizer with gradient clipping and EMA update."""
    self.scaler.unscale_(self.optimizer)
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
    self.scaler.step(self.optimizer)
    self.scaler.update()
    self.optimizer.zero_grad()
    if self.ema:
        self.ema.update(self.model)
