def __init__(self, args, cfg, device):
    self.args = args
    self.cfg = cfg
    self.device = device
    self.max_epoch = args.epochs
    if args.resume:
        self.ckpt = torch.load(args.resume, map_location='cpu')
    self.rank = args.rank
    self.local_rank = args.local_rank
    self.world_size = args.world_size
    self.main_process = self.rank in [-1, 0]
    self.save_dir = args.save_dir
    self.data_dict = load_yaml(args.data_path)
    self.num_classes = self.data_dict['nc']
    self.distill_ns = True if self.args.distill and self.cfg.model.type in [
        'YOLOv6n', 'YOLOv6s'] else False
    model = self.get_model(args, cfg, self.num_classes, device)
    if self.args.distill:
        if self.args.fuse_ab:
            LOGGER.error(
                'ERROR in: Distill models should turn off the fuse_ab.\n')
            exit()
        self.teacher_model = self.get_teacher_model(args, cfg, self.
            num_classes, device)
    if self.args.quant:
        self.quant_setup(model, cfg, device)
    if cfg.training_mode == 'repopt':
        scales = self.load_scale_from_pretrained_models(cfg, device)
        reinit = False if cfg.model.pretrained is not None else True
        self.optimizer = RepVGGOptimizer(model, scales, args, cfg, reinit=
            reinit)
    else:
        self.optimizer = self.get_optimizer(args, cfg, model)
    self.scheduler, self.lf = self.get_lr_scheduler(args, cfg, self.optimizer)
    self.ema = ModelEMA(model) if self.main_process else None
    self.tblogger = SummaryWriter(self.save_dir) if self.main_process else None
    self.start_epoch = 0
    if hasattr(self, 'ckpt'):
        resume_state_dict = self.ckpt['model'].float().state_dict()
        model.load_state_dict(resume_state_dict, strict=True)
        self.start_epoch = self.ckpt['epoch'] + 1
        self.optimizer.load_state_dict(self.ckpt['optimizer'])
        self.scheduler.load_state_dict(self.ckpt['scheduler'])
        if self.main_process:
            self.ema.ema.load_state_dict(self.ckpt['ema'].float().state_dict())
            self.ema.updates = self.ckpt['updates']
        if self.start_epoch > self.max_epoch - self.args.stop_aug_last_n_epoch:
            self.cfg.data_aug.mosaic = 0.0
            self.cfg.data_aug.mixup = 0.0
    self.train_loader, self.val_loader = self.get_data_loader(self.args,
        self.cfg, self.data_dict)
    self.model = self.parallel_model(args, model, device)
    self.model.nc, self.model.names = self.data_dict['nc'], self.data_dict[
        'names']
    self.max_stepnum = len(self.train_loader)
    self.batch_size = args.batch_size
    self.img_size = args.img_size
    self.rect = args.rect
    self.vis_imgs_list = []
    self.write_trainbatch_tb = args.write_trainbatch_tb
    self.color = [tuple(np.random.choice(range(256), size=3)) for _ in
        range(self.model.nc)]
    self.specific_shape = args.specific_shape
    self.height = args.height
    self.width = args.width
    self.loss_num = 3
    self.loss_info = ['Epoch', 'lr', 'iou_loss', 'dfl_loss', 'cls_loss']
    if self.args.distill:
        self.loss_num += 1
        self.loss_info += ['cwd_loss']
