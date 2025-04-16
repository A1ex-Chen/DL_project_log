def final_eval(self):
    """Evaluate trained model and save validation results."""
    for f in (self.last, self.best):
        if f.exists():
            strip_optimizer(f)
            if f is self.best:
                LOGGER.info(f'\nValidating {f}...')
                self.validator.args.data = self.args.data
                self.validator.args.plots = self.args.plots
                self.metrics = self.validator(model=f)
                self.metrics.pop('fitness', None)
                self.run_callbacks('on_fit_epoch_end')
    LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
