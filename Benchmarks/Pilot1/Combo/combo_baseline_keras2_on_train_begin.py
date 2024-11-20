def on_train_begin(self, logs={}):
    self.val_losses = []
    self.best_val_loss = np.Inf
    self.best_model = None
