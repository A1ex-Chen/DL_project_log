def on_train_end(self):
    """Finalize training process"""
    for callback in self.callbacks:
        callback.on_train_end(self)
