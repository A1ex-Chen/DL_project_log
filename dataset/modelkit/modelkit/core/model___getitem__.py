def __getitem__(self, key: str) ->Union['Model', 'AsyncModel',
    'WrappedAsyncModel']:
    return self.models[key]
