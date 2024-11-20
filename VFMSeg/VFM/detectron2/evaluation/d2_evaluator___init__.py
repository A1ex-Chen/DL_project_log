def __init__(self, *, dataset_name, prefix='', **kwargs):
    super().__init__(dataset_name=dataset_name, **kwargs)
    self.dataset_name = dataset_name
    if len(prefix) and not prefix.endswith('_'):
        prefix += '_'
    self.prefix = prefix
