def on_train_begin(self):
    """Start the training process - always used, even in restarts"""
    for callback in self.callbacks:
        callback.on_train_begin(self)
