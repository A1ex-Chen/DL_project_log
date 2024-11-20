@cached_property
def model(self):
    return MarianMTModel(self.config)
