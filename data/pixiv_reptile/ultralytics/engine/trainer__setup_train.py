def _setup_train(self, world_size):
    """Builds dataloaders and optimizer on correct rank process."""
    self.run_callbacks('on_pretrain_routine_start')
    ckpt = self.setup_model()
    self.model = self.model.to(self.device)
    self.set_model_attributes()
    freeze_list = self.args.freeze if isinstance(self.args.freeze, list
        ) else range(self.args.freeze) if isinstance(self.args.freeze, int
        ) else []
    always_freeze_names = ['.dfl']
    freeze_layer_names = [f'model.{x}.' for x in freeze_list
        ] + always_freeze_names
    for k, v in self.model.named_parameters():
        if any(x in k for x in freeze_layer_names):
            LOGGER.info(f"Freezing layer '{k}'")
            v.requires_grad = False
        elif not v.requires_grad and v.dtype.is_floating_point:
            LOGGER.info(
                f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. See ultralytics.engine.trainer for customization of frozen layers."
                )
            v.requires_grad = True
    self.amp = torch.tensor(self.args.amp).to(self.device)
    if self.amp and RANK in {-1, 0}:
        callbacks_backup = callbacks.default_callbacks.copy()
        self.amp = torch.tensor(check_amp(self.model), device=self.device)
        callbacks.default_callbacks = callbacks_backup
    if RANK > -1 and world_size > 1:
        dist.broadcast(self.amp, src=0)
    self.amp = bool(self.amp)
    self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
    if world_size > 1:
        self.model = nn.parallel.DistributedDataParallel(self.model,
            device_ids=[RANK], find_unused_parameters=True)
    gs = max(int(self.model.stride.max() if hasattr(self.model, 'stride') else
        32), 32)
    self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs,
        max_dim=1)
    self.stride = gs
    if self.batch_size < 1 and RANK == -1:
        self.args.batch = self.batch_size = check_train_batch_size(model=
            self.model, imgsz=self.args.imgsz, amp=self.amp, batch=self.
            batch_size)
    batch_size = self.batch_size // max(world_size, 1)
    self.train_loader = self.get_dataloader(self.trainset, batch_size=
        batch_size, rank=RANK, mode='train')
    if RANK in {-1, 0}:
        self.test_loader = self.get_dataloader(self.testset, batch_size=
            batch_size if self.args.task == 'obb' else batch_size * 2, rank
            =-1, mode='val')
        self.validator = self.get_validator()
        metric_keys = self.validator.metrics.keys + self.label_loss_items(
            prefix='val')
        self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
        self.ema = ModelEMA(self.model)
        if self.args.plots:
            self.plot_training_labels()
    self.accumulate = max(round(self.args.nbs / self.batch_size), 1)
    weight_decay = (self.args.weight_decay * self.batch_size * self.
        accumulate / self.args.nbs)
    iterations = math.ceil(len(self.train_loader.dataset) / max(self.
        batch_size, self.args.nbs)) * self.epochs
    self.optimizer = self.build_optimizer(model=self.model, name=self.args.
        optimizer, lr=self.args.lr0, momentum=self.args.momentum, decay=
        weight_decay, iterations=iterations)
    self._setup_scheduler()
    self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
    self.resume_training(ckpt)
    self.scheduler.last_epoch = self.start_epoch - 1
    self.run_callbacks('on_pretrain_routine_end')
