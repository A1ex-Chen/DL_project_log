def process(self, tp, conf, pred_cls, target_cls):
    """Process predicted results for object detection and update metrics."""
    results = ap_per_class(tp, conf, pred_cls, target_cls, plot=self.plot,
        save_dir=self.save_dir, names=self.names, on_plot=self.on_plot)[2:]
    self.box.nc = len(self.names)
    self.box.update(results)
