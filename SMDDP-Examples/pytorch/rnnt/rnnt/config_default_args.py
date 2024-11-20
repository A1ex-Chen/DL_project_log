def default_args(klass):
    sig = inspect.signature(klass.__init__)
    return {k: v.default for k, v in sig.parameters.items() if k != 'self'}
