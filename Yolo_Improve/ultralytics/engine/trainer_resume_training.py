def resume_training(self, ckpt):
    """Resume YOLO training from given epoch and best fitness."""
    if ckpt is None or not self.resume:
        return
    best_fitness = 0.0
    start_epoch = ckpt.get('epoch', -1) + 1
    if ckpt.get('optimizer', None) is not None:
        self.optimizer.load_state_dict(ckpt['optimizer'])
        best_fitness = ckpt['best_fitness']
    if self.ema and ckpt.get('ema'):
        self.ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
        self.ema.updates = ckpt['updates']
    assert start_epoch > 0, f"""{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.
Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"""
    LOGGER.info(
        f'Resuming training {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs'
        )
    if self.epochs < start_epoch:
        LOGGER.info(
            f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
        self.epochs += ckpt['epoch']
    self.best_fitness = best_fitness
    self.start_epoch = start_epoch
    if start_epoch > self.epochs - self.args.close_mosaic:
        self._close_dataloader_mosaic()
