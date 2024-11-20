def on_train_begin(self, logs=None):
    super(MyProgbarLogger, self).on_train_begin(logs)
    self.verbose = 1
    self.extra_log_values = []
    self.params['samples'] = self.samples
    self.params['metrics'] = []
