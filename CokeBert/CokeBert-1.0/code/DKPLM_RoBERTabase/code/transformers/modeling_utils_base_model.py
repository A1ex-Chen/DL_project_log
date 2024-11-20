@property
def base_model(self):
    return getattr(self, self.base_model_prefix, self)
