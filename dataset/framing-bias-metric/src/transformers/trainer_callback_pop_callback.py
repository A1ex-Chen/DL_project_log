def pop_callback(self, callback):
    if isinstance(callback, type):
        for cb in self.callbacks:
            if isinstance(cb, callback):
                self.callbacks.remove(cb)
                return cb
    else:
        for cb in self.callbacks:
            if cb == callback:
                self.callbacks.remove(cb)
                return cb
