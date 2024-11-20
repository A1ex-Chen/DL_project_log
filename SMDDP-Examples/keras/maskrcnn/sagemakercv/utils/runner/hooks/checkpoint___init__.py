def __init__(self, interval=-1, save_optimizer=True, out_dir=None,
    backbone_checkpoint=None, **kwargs):
    self.interval = interval
    self.save_optimizer = save_optimizer
    self.out_dir = out_dir
    self.backbone_checkpoint = backbone_checkpoint
    self.args = kwargs
