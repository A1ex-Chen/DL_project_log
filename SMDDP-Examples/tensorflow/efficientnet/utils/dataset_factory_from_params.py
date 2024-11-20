@classmethod
def from_params(cls, *args, **kwargs):
    """Construct a dataset builder from a default config and any overrides."""
    config = DatasetConfig.from_args(*args, **kwargs)
    return cls(config)
