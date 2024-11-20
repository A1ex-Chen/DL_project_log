def __init__(self, settings: Union[Dict, AzureStorageDriverSettings],
    client: Optional[BlobServiceClient]=None):
    if isinstance(settings, dict):
        settings = AzureStorageDriverSettings(**settings)
    if not (client or settings.connection_string):
        raise ValueError(
            'Connection string needs to be set for Azure storage driver')
    super().__init__(settings, client, client_configuration={
        'connection_string': settings.connection_string})
