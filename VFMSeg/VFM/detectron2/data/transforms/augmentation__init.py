def _init(self, params=None):
    if params:
        for k, v in params.items():
            if k != 'self' and not k.startswith('_'):
                setattr(self, k, v)
