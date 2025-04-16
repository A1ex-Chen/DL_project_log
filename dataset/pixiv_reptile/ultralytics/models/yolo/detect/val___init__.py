def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None,
    _callbacks=None):
    """Initialize detection model with necessary variables and settings."""
    super().__init__(dataloader, save_dir, pbar, args, _callbacks)
    self.nt_per_class = None
    self.nt_per_image = None
    self.is_coco = False
    self.is_lvis = False
    self.class_map = None
    self.args.task = 'detect'
    self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
    self.iouv = torch.linspace(0.5, 0.95, 10)
    self.niou = self.iouv.numel()
    self.lb = []
