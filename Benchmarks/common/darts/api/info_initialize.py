def initialize(self):
    for callback in self.callbacks:
        callback.on_initialization(self)
