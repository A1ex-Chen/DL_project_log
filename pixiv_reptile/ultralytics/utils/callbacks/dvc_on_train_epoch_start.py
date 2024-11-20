def on_train_epoch_start(trainer):
    """Sets the global variable _training_epoch value to True at the start of training each epoch."""
    global _training_epoch
    _training_epoch = True
