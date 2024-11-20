def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
    """
        Initializes the BaseTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
    self.args = get_cfg(cfg, overrides)
    self.check_resume(overrides)
    self.device = select_device(self.args.device, self.args.batch)
    self.validator = None
    self.metrics = None
    self.plots = {}
    init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic
        )
    self.save_dir = get_save_dir(self.args)
    self.args.name = self.save_dir.name
    self.wdir = self.save_dir / 'weights'
    if RANK in {-1, 0}:
        self.wdir.mkdir(parents=True, exist_ok=True)
        self.args.save_dir = str(self.save_dir)
        yaml_save(self.save_dir / 'args.yaml', vars(self.args))
    self.last, self.best = self.wdir / 'last.pt', self.wdir / 'best.pt'
    self.save_period = self.args.save_period
    self.batch_size = self.args.batch
    self.epochs = self.args.epochs
    self.start_epoch = 0
    if RANK == -1:
        print_args(vars(self.args))
    if self.device.type in {'cpu', 'mps'}:
        self.args.workers = 0
    self.model = check_model_file_from_stem(self.args.model)
    with torch_distributed_zero_first(RANK):
        self.trainset, self.testset = self.get_dataset()
    self.ema = None
    self.lf = None
    self.scheduler = None
    self.best_fitness = None
    self.fitness = None
    self.loss = None
    self.tloss = None
    self.loss_names = ['Loss']
    self.csv = self.save_dir / 'results.csv'
    self.plot_idx = [0, 1, 2]
    self.hub_session = None
    self.callbacks = _callbacks or callbacks.get_default_callbacks()
    if RANK in {-1, 0}:
        callbacks.add_integration_callbacks(self)
