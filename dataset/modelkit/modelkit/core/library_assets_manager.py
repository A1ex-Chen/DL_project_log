@property
def assets_manager(self):
    if self._assets_manager is None:
        logger.info('Instantiating AssetsManager', lazy_loading=self.
            _lazy_loading)
        self._assets_manager = AssetsManager(**self.assetsmanager_settings)
    return self._assets_manager
