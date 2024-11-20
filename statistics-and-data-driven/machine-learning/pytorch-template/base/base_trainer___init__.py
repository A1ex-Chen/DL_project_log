def __init__(self, model, criterion, metric_ftns, optimizer, config):
    self.config = config
    self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
    self.model = model
    self.criterion = criterion
    self.metric_ftns = metric_ftns
    self.optimizer = optimizer
    cfg_trainer = config['trainer']
    self.epochs = cfg_trainer['epochs']
    self.save_period = cfg_trainer['save_period']
    self.monitor = cfg_trainer.get('monitor', 'off')
    if self.monitor == 'off':
        self.mnt_mode = 'off'
        self.mnt_best = 0
    else:
        self.mnt_mode, self.mnt_metric = self.monitor.split()
        assert self.mnt_mode in ['min', 'max']
        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = cfg_trainer.get('early_stop', inf)
        if self.early_stop <= 0:
            self.early_stop = inf
    self.start_epoch = 1
    self.checkpoint_dir = config.save_dir
    self.writer = TensorboardWriter(config.log_dir, self.logger,
        cfg_trainer['tensorboard'])
    if config.resume is not None:
        self._resume_checkpoint(config.resume)
