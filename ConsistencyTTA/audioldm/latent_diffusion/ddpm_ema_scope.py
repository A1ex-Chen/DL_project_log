@contextmanager
def ema_scope(self, context=None):
    if self.use_ema:
        self.model_ema.store(self.model.parameters())
        self.model_ema.copy_to(self.model)
        if context is not None:
            pass
    try:
        yield None
    finally:
        if self.use_ema:
            self.model_ema.restore(self.model.parameters())
            if context is not None:
                pass
