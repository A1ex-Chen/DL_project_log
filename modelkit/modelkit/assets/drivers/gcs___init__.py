def __init__(self, settings: Union[Dict, GCSStorageDriverSettings], client:
    Optional[Client]=None):
    if isinstance(settings, dict):
        settings = GCSStorageDriverSettings(**settings)
    super().__init__(settings=settings, client=client, client_configuration
        ={'service_account_path': settings.service_account_path})
