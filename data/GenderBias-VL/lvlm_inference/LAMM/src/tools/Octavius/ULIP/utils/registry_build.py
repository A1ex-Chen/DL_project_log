def build(self, *args, **kwargs):
    return self.build_func(*args, **kwargs, registry=self)
