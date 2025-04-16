def __init__(self, settings: Union[Dict, StorageDriverSettings], client:
    Optional[Any]=None, client_configuration: Optional[Dict[str, Any]]=None
    ) ->None:
    if isinstance(settings, dict):
        settings = StorageDriverSettings(**settings)
    self._client = client
    self.client_configuration = client_configuration or {}
    self.bucket = settings.bucket
    self.lazy_driver = settings.lazy_driver
    if not (self.lazy_driver or client):
        self._client = self.build_client(self.client_configuration)
