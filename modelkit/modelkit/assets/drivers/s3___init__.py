def __init__(self, settings: Union[Dict, S3StorageDriverSettings], client:
    Optional[boto3.client]=None):
    if isinstance(settings, dict):
        settings = S3StorageDriverSettings(**settings)
    client_configuration = {'endpoint_url': settings.s3_endpoint,
        'aws_access_key_id': settings.aws_access_key_id,
        'aws_secret_access_key': settings.aws_secret_access_key,
        'region_name': settings.aws_default_region, 'aws_session_token':
        settings.aws_session_token}
    self.aws_kms_key_id = settings.aws_kms_key_id
    super().__init__(settings=settings, client=client, client_configuration
        =client_configuration)
