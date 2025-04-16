def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    warnings.warn(DEPRECATION_WARNING.format('processor'), FutureWarning)
