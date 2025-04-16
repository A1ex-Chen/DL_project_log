def __init__(self, args, model, train_loader, val_loader, test_loader,
    export_root):
    self.args = args
    self.device = args.device
    self.model = model.to(self.device)
    self.is_parallel = args.num_gpu > 1
    if self.is_parallel:
        self.model = nn.DataParallel(self.model)
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.test_loader = test_loader
    self.optimizer = self._create_optimizer()
    if args.enable_lr_schedule:
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer,
            step_size=args.decay_step, gamma=args.gamma)
    self.num_epochs = args.num_epochs
    self.metric_ks = args.metric_ks
    self.best_metric = args.best_metric
    self.export_root = export_root
    self.writer, self.train_loggers, self.val_loggers = self._create_loggers()
    self.add_extra_loggers()
    self.logger_service = LoggerService(self.train_loggers, self.val_loggers)
    self.log_period_as_iter = args.log_period_as_iter
