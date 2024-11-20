def on_train_end(self, logs={}):
    self.model.save_weights(self.fname)
