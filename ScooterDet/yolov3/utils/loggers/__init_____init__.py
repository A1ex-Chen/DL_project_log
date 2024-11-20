def __init__(self, opt, console_logger, include=('tb', 'wandb')):
    self.save_dir = Path(opt.save_dir)
    self.include = include
    self.console_logger = console_logger
    self.csv = self.save_dir / 'results.csv'
    if 'tb' in self.include:
        prefix = colorstr('TensorBoard: ')
        self.console_logger.info(
            f"{prefix}Start with 'tensorboard --logdir {self.save_dir.parent}', view at http://localhost:6006/"
            )
        self.tb = SummaryWriter(str(self.save_dir))
    if wandb and 'wandb' in self.include:
        self.wandb = wandb.init(project=web_project_name(str(opt.project)),
            name=None if opt.name == 'exp' else opt.name, config=opt)
    else:
        self.wandb = None
