def __init__(self, save_dir=None, weights=None, opt=None, hyp=None, logger=
    None, include=LOGGERS):
    self.save_dir = save_dir
    self.weights = weights
    self.opt = opt
    self.hyp = hyp
    self.logger = logger
    self.include = include
    self.keys = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',
        'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5',
        'metrics/mAP_0.5:0.95', 'val/box_loss', 'val/obj_loss',
        'val/cls_loss', 'x/lr0', 'x/lr1', 'x/lr2']
    self.best_keys = ['best/epoch', 'best/precision', 'best/recall',
        'best/mAP_0.5', 'best/mAP_0.5:0.95']
    for k in LOGGERS:
        setattr(self, k, None)
    self.csv = True
    if not wandb:
        prefix = colorstr('Weights & Biases: ')
        s = (
            f"{prefix}run 'pip install wandb' to automatically track and visualize YOLOv5 🚀 runs in Weights & Biases"
            )
        self.logger.info(s)
    if not clearml:
        prefix = colorstr('ClearML: ')
        s = (
            f"{prefix}run 'pip install clearml' to automatically track, visualize and remotely train YOLOv5 🚀 runs in ClearML"
            )
        self.logger.info(s)
    s = self.save_dir
    if 'tb' in self.include and not self.opt.evolve:
        prefix = colorstr('TensorBoard: ')
        self.logger.info(
            f"{prefix}Start with 'tensorboard --logdir {s.parent}', view at http://localhost:6006/"
            )
        self.tb = SummaryWriter(str(s))
    if wandb and 'wandb' in self.include:
        wandb_artifact_resume = isinstance(self.opt.resume, str
            ) and self.opt.resume.startswith('wandb-artifact://')
        run_id = torch.load(self.weights).get('wandb_id'
            ) if self.opt.resume and not wandb_artifact_resume else None
        self.opt.hyp = self.hyp
        self.wandb = WandbLogger(self.opt, run_id)
        if pkg.parse_version(wandb.__version__) >= pkg.parse_version('0.12.11'
            ):
            s = (
                'YOLOv5 temporarily requires wandb version 0.12.10 or below. Some features may not work as expected.'
                )
            self.logger.warning(s)
    else:
        self.wandb = None
    if clearml and 'clearml' in self.include:
        self.clearml = ClearmlLogger(self.opt, self.hyp)
    else:
        self.clearml = None
