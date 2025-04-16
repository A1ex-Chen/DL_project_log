def init_metrics(self, model):
    """Initialize metrics and select mask processing function based on save_json flag."""
    super().init_metrics(model)
    self.plot_masks = []
    if self.args.save_json:
        check_requirements('pycocotools>=2.0.6')
        self.process = ops.process_mask_upsample
    else:
        self.process = ops.process_mask
    self.stats = dict(tp_m=[], tp=[], conf=[], pred_cls=[], target_cls=[],
        target_img=[])
