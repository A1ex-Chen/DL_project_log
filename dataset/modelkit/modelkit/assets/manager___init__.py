def __init__(self, assets_dir: Optional[str]=None, timeout: Optional[int]=
    None, storage_provider: Optional[StorageProvider]=None):
    self.assets_dir = assets_dir or os.environ.get('MODELKIT_ASSETS_DIR'
        ) or os.getcwd()
    if not os.path.isdir(self.assets_dir):
        raise FileNotFoundError(
            f'Assets directory {self.assets_dir} does not exist')
    self.timeout: int = int(timeout or os.environ.get(
        'MODELKIT_ASSETS_TIMEOUT_S') or 10)
    self.storage_provider = storage_provider
    if not self.storage_provider:
        try:
            self.storage_provider = StorageProvider()
            logger.debug('AssetsManager created with remote storage provider',
                driver=self.storage_provider.driver)
        except NoConfiguredProviderError:
            logger.info('No remote storage provider configured')
    if self.storage_provider and isinstance(self.storage_provider.driver,
        LocalStorageDriver) and self.assets_dir == os.path.join(self.
        storage_provider.driver.bucket, self.storage_provider.prefix):
        raise errors.StorageDriverError(
            'Incompatible configuration: LocalStorageDriver and AssetDir are pointing to the same folder. If assets are already downloaded,consider removing storage provider configuration.'
            )
