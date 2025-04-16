@property
def components(self) ->Dict[str, Any]:
    return {k: getattr(self, k) for k in self.config.keys() if not k.
        startswith('_')}
