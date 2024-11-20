def __init__(self, settings: Union[Dict, LocalStorageDriverSettings]):
    if isinstance(settings, dict):
        settings = LocalStorageDriverSettings(**settings)
    super().__init__(settings)
    if not os.path.isdir(self.bucket):
        raise FileNotFoundError
