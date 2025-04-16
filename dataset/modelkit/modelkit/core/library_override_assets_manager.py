@property
def override_assets_manager(self):
    if not self.settings.override_assets_dir:
        return None
    if self._override_assets_manager is None:
        logger.info('Instantiating Override AssetsManager', lazy_loading=
            self._lazy_loading)
        self._override_assets_manager = AssetsManager(assets_dir=self.
            settings.override_assets_dir)
        self._override_assets_manager.storage_provider = None
    return self._override_assets_manager
