def on_train_begin(self, logs={}):
    logs = logs or {}
    if self.clr_iterations == 0:
        K.set_value(self.model.optimizer.lr, self.base_lr)
    else:
        K.set_value(self.model.optimizer.lr, self.clr())
