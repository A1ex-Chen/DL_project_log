def compile(self, optimizer, run_eagerly=True):
    super(TwoStageDetector, self).compile(run_eagerly=run_eagerly)
    self.optimizer = optimizer
