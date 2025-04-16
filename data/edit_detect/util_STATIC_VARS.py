@property
def STATIC_VARS(self):
    return [attr for attr in dir(ModelDataset) if not callable(getattr(
        ModelDataset, attr)) and not attr.startswith('__')]
