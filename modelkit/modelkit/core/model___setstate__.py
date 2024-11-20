def __setstate__(self, state):
    self.__dict__ = state
    self.initialize_validation_models()
