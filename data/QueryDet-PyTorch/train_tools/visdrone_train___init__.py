def __init__(self, cfg, resume=False, reuse_ckpt=False):
    """
        Args:
            cfg (CfgNode):
        """
    super(DefaultTrainer, self).__init__()
    logger = logging.getLogger('detectron2')
    if not logger.isEnabledFor(logging.INFO):
        setup_logger()
    cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
    model = self.build_model(cfg)
    ckpt = DetectionCheckpointer(model)
    self.start_iter = 0
    self.start_iter = ckpt.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume
        ).get('iteration', -1) + 1
    self.iter = self.start_iter
    optimizer = self.build_optimizer(cfg, model)
    data_loader = self.build_train_loader(cfg)
    if comm.get_world_size() > 1:
        model = DistributedDataParallel(model, device_ids=[comm.
            get_local_rank()], broadcast_buffers=False)
    self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
        model, data_loader, optimizer)
    self.scheduler = self.build_lr_scheduler(cfg, optimizer)
    self.checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR,
        optimizer=optimizer, scheduler=self.scheduler)
    self.start_iter = 0
    self.max_iter = cfg.SOLVER.MAX_ITER
    self.cfg = cfg
    self.register_hooks(self.build_hooks())
