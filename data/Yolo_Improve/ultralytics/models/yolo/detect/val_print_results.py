def print_results(self):
    """Prints training/validation set metrics per class."""
    pf = '%22s' + '%11i' * 2 + '%11.3g' * len(self.metrics.keys)
    LOGGER.info(pf % ('all', self.seen, self.nt_per_class.sum(), *self.
        metrics.mean_results()))
    if self.nt_per_class.sum() == 0:
        LOGGER.warning(
            f'WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels'
            )
    if self.args.verbose and not self.training and self.nc > 1 and len(self
        .stats):
        for i, c in enumerate(self.metrics.ap_class_index):
            LOGGER.info(pf % (self.names[c], self.nt_per_image[c], self.
                nt_per_class[c], *self.metrics.class_result(i)))
    if self.args.plots:
        for normalize in (True, False):
            self.confusion_matrix.plot(save_dir=self.save_dir, names=self.
                names.values(), normalize=normalize, on_plot=self.on_plot)
