def init_metrics(self, model):
    """Initialize evaluation metrics for YOLO."""
    super().init_metrics(model)
    val = self.data.get(self.args.split, '')
    self.is_dota = isinstance(val, str) and 'DOTA' in val
