def on_epoch_end(self, epoch, logs={}):
    val_loss = logs.get('val_loss')
    self.val_losses.append(val_loss)
    if val_loss < self.best_val_loss:
        self.best_model = keras.models.clone_model(self.model)
        self.best_val_loss = val_loss
