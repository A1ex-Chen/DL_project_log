def __init__(self, logger: Logger, config: Config, model: Model, optimizer:
    Optimizer, scheduler: _LRScheduler, terminator: Event, val_evaluator:
    Evaluator, test_evaluator: Optional[Evaluator], db: DB,
    path_to_checkpoints_dir: str):
    super().__init__()
    self.logger = logger
    self.config = config
    self.model = model
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.terminator = terminator
    self.val_evaluator = val_evaluator
    self.test_evaluator = test_evaluator
    self.db = db
    self.path_to_checkpoints_dir = path_to_checkpoints_dir
    self.summary_writer = SummaryWriter(log_dir=os.path.join(
        path_to_checkpoints_dir, 'tensorboard'))
    self.losses = deque(maxlen=100)
    self.time_checkpoint = None
    self.batches_counter = None
    self.best_epoch = None
    self.best_accuracy = -1
    self.checkpoint_infos = []
    checkpoints = db.select_checkpoint_table()
    for available_checkpoint in [checkpoint for checkpoint in checkpoints if
        checkpoint.is_available]:
        assert available_checkpoint.metrics.specific['categories'][0] == 'mean'
        checkpoint_info = Trainer.Callback.CheckpointInfo(available_checkpoint
            .epoch, available_checkpoint.avg_loss, available_checkpoint.
            metrics.overall['accuracy'])
        self.checkpoint_infos.append(checkpoint_info)
    best_checkpoints = [checkpoint for checkpoint in checkpoints if
        checkpoint.is_best]
    assert len(best_checkpoints) <= 1
    if len(best_checkpoints) > 0:
        best_checkpoint = best_checkpoints[0]
        self.best_epoch = best_checkpoint.epoch
        self.best_accuracy = best_checkpoint.metrics.overall['accuracy']
    self.profile_enabled = True
