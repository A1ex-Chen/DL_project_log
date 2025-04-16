def on_batch_end(self, epoch, logs=None):
    logs = logs or {}
    self.trn_iterations += 1
    self.clr_iterations += 1
    K.set_value(self.model.optimizer.lr, self.clr())
    self.history.setdefault('lr', []).append(K.get_value(self.model.
        optimizer.lr))
    self.history.setdefault('iterations', []).append(self.trn_iterations)
    for k, v in logs.items():
        self.history.setdefault(k, []).append(v)
