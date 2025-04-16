def finalize_metrics(self, *args, **kwargs):
    """Finalizes metrics of the model such as confusion_matrix and speed."""
    self.confusion_matrix.process_cls_preds(self.pred, self.targets)
    if self.args.plots:
        for normalize in (True, False):
            self.confusion_matrix.plot(save_dir=self.save_dir, names=self.
                names.values(), normalize=normalize, on_plot=self.on_plot)
    self.metrics.speed = self.speed
    self.metrics.confusion_matrix = self.confusion_matrix
    self.metrics.save_dir = self.save_dir
