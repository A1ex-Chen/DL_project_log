@staticmethod
def build_client(client_configuration: Dict[str, str]) ->boto3.client:
    return boto3.client('s3', **client_configuration)
