def sample(self, *args, **kwds):
    ret = super().sample(*args, **kwds)
    self.empty_buffer()
    self.reset_infer_state()
    return ret
