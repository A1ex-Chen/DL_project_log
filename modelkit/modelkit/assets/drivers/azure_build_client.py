@staticmethod
def build_client(client_configuration: Dict[str, str]) ->BlobServiceClient:
    connection_string = client_configuration.get('connection_string')
    if not connection_string:
        raise ValueError(
            'Connection string needs to be set for Azure storage driver')
    return BlobServiceClient.from_connection_string(connection_string)
