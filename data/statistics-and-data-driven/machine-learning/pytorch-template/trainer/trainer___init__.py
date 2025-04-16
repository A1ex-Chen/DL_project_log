def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
    data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
    super().__init__(model, criterion, metric_ftns, optimizer, config)
    self.config = config
    self.device = device
    self.data_loader = data_loader
    if len_epoch is None:
        self.len_epoch = len(self.data_loader)
    else:
        self.data_loader = inf_loop(data_loader)
        self.len_epoch = len_epoch
    self.valid_data_loader = valid_data_loader
    self.do_validation = self.valid_data_loader is not None
    self.lr_scheduler = lr_scheduler
    self.log_step = int(np.sqrt(data_loader.batch_size))
    self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.
        metric_ftns], writer=self.writer)
    self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.
        metric_ftns], writer=self.writer)
