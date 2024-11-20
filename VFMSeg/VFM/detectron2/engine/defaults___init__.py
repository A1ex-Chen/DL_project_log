def __init__(self, cfg):
    """
        Args:
            cfg (CfgNode):
        """
    super().__init__()
    logger = logging.getLogger('detectron2')
    if not logger.isEnabledFor(logging.INFO):
        setup_logger()
    cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
    model = self.build_model(cfg)
    optimizer = self.build_optimizer(cfg, model)
    data_loader = self.build_train_loader(cfg)
    model = create_ddp_model(model, broadcast_buffers=False,
        find_unused_parameters=cfg.FIND_UNUSED_PARAMETERS)
    self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
        model, data_loader, optimizer)
    self.scheduler = self.build_lr_scheduler(cfg, optimizer)
    self.checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR,
        trainer=weakref.proxy(self))
    self.start_iter = 0
    self.max_iter = cfg.SOLVER.MAX_ITER
    self.cfg = cfg
    self.register_hooks(self.build_hooks())
