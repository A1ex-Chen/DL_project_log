def call_model_init(self, trial=None):
    model_init_argcount = len(inspect.signature(self.model_init).parameters)
    if model_init_argcount == 0:
        model = self.model_init()
    elif model_init_argcount == 1:
        model = self.model_init(trial)
    else:
        raise RuntimeError('model_init should have 0 or 1 argument.')
    if model is None:
        raise RuntimeError('model_init should not return None.')
    return model
