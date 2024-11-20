def add_callback(self, callback):
    cb = callback() if isinstance(callback, type) else callback
    cb_class = callback if isinstance(callback, type) else callback.__class__
    if cb_class in [c.__class__ for c in self.callbacks]:
        logger.warn(
            f'You are adding a {cb_class} to the callbacks of this Trainer, but there is already one. The current'
             + """list of callbacks is
:""" + self.callback_list)
    self.callbacks.append(cb)
