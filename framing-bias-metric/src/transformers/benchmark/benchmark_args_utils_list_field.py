def list_field(default=None, metadata=None):
    return field(default_factory=lambda : default, metadata=metadata)
