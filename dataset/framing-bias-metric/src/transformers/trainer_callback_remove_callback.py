def remove_callback(self, callback):
    if isinstance(callback, type):
        for cb in self.callbacks:
            if isinstance(cb, callback):
                self.callbacks.remove(cb)
                return
    else:
        self.callbacks.remove(callback)
