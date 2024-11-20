def __getattr__(cls, key):
    if key.startswith('_'):
        return super().__getattr__(cls, key)
    requires_backends(cls, cls._backends)
