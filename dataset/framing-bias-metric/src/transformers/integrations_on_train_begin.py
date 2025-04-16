def on_train_begin(self, args, state, control, model=None, **kwargs):
    if not self._initialized:
        self.setup(args, state, model)
