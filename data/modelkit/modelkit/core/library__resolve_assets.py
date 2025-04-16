def _resolve_assets(self, model_name):
    """
        This function fetches assets for the current model and its dependent models
        and populates the assets_info dictionary with the paths.
        """
    logger.debug('Resolving asset for Model', model_name=model_name)
    configuration = self.configuration[model_name]
    for dep_name in configuration.model_dependencies.values():
        self._resolve_assets(dep_name)
    if not configuration.asset:
        return
    model_settings = {**configuration.model_settings, **self.
        required_models.get(model_name, {})}
    if 'asset_path' in model_settings:
        asset_path = model_settings.pop('asset_path')
        logger.debug('Overriding asset from Model settings', model_name=
            model_name, asset_path=asset_path)
        self.assets_info[configuration.asset] = AssetInfo(path=asset_path)
    asset_spec = AssetSpec.from_string(configuration.asset)
    venv = 'MODELKIT_{}_FILE'.format(re.sub('[\\/\\-\\.]+', '_', asset_spec
        .name).upper())
    local_file = os.environ.get(venv)
    if local_file:
        logger.debug('Overriding asset from environment variable',
            asset_name=asset_spec.name, path=local_file)
        self.assets_info[configuration.asset] = AssetInfo(path=local_file)
    venv = 'MODELKIT_{}_VERSION'.format(re.sub('[\\/\\-\\.]+', '_',
        asset_spec.name).upper())
    version = os.environ.get(venv)
    if version:
        logger.debug('Overriding asset version from environment variable',
            asset_name=asset_spec.name, path=local_file)
        asset_spec = AssetSpec.from_string(asset_spec.name + ':' + version)
    if self.override_assets_manager:
        try:
            self.assets_info[configuration.asset] = AssetInfo(**self.
                override_assets_manager.fetch_asset(spec=AssetSpec(name=
                asset_spec.name, sub_part=asset_spec.sub_part), return_info
                =True))
            logger.debug('Asset has been overriden', name=asset_spec.name)
        except modelkit.assets.errors.AssetDoesNotExistError:
            logger.debug('Asset not found in overriden prefix', name=
                asset_spec.name)
    if configuration.asset not in self.assets_info:
        self.assets_info[configuration.asset] = AssetInfo(**self.
            assets_manager.fetch_asset(asset_spec, return_info=True))
