def __init__(self, timeout_s: Optional[int]=None, prefix: Optional[str]=
    None, force_download: Optional[bool]=None, provider: Optional[str]=None,
    client: Optional[Any]=None, **driver_settings):
    self.timeout = timeout_s or int(os.environ.get(
        'MODELKIT_STORAGE_TIMEOUT_S', 300))
    self.prefix = prefix or os.environ.get('MODELKIT_STORAGE_PREFIX'
        ) or 'modelkit-assets'
    self.force_download = force_download or bool(os.environ.get(
        'MODELKIT_STORAGE_FORCE_DOWNLOAD'))
    provider = provider or os.environ.get('MODELKIT_STORAGE_PROVIDER')
    if not provider:
        raise NoConfiguredProviderError()
    if provider == 'gcs':
        if not has_gcs:
            raise DriverNotInstalledError(
                'GCS driver not installed, install modelkit[assets-gcs]')
        gcs_driver_settings = GCSStorageDriverSettings(**driver_settings)
        self.driver = GCSStorageDriver(gcs_driver_settings, client)
    elif provider == 's3':
        if not has_s3:
            raise DriverNotInstalledError(
                'S3 driver not installed, install modelkit[assets-s3]')
        s3_driver_settings = S3StorageDriverSettings(**driver_settings)
        self.driver = S3StorageDriver(s3_driver_settings, client)
    elif provider == 'local':
        local_driver_settings = LocalStorageDriverSettings(**driver_settings)
        self.driver = LocalStorageDriver(local_driver_settings)
    elif provider == 'az':
        if not has_az:
            raise DriverNotInstalledError(
                'Azure driver not installed, install modelkit[assets-az]')
        az_driver_settings = AzureStorageDriverSettings(**driver_settings)
        self.driver = AzureStorageDriver(az_driver_settings, client)
    else:
        raise UnknownDriverError()
