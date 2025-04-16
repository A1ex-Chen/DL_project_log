def __init__(self, trainer, cfg, work_dir=None, log_level=logging.INFO,
    logger=None, name=None):
    self.trainer = trainer
    self.cfg = cfg
    self.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    if work_dir is not None:
        self.work_dir = os.path.abspath(work_dir)
    else:
        self.work_dir = self.cfg.PATHS.OUT_DIR
    os.makedirs(self.work_dir, exist_ok=True)
    self.init_tensorboard()
    if name is None:
        self._model_name = (self.trainer.model.__class__.__name__ + ' ' +
            self.timestamp)
    self._rank, self._local_rank, self._size, self._local_size = get_dist_info(
        )
    if logger is None:
        self.logger = self.init_logger(work_dir, log_level)
    else:
        self.logger = logger
    self.log_buffer = LogBuffer()
    self.mode = None
    self._hooks = []
    self._epoch = 0
    self._iter = 0
    self._inner_iter = 0
