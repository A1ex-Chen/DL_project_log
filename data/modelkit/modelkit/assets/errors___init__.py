def __init__(self, name, version, local_versions):
    super().__init__(
        f'Asset version `{version}` for `{name}` does not exist locally. Available asset versions: '
         + ', '.join(local_versions))
