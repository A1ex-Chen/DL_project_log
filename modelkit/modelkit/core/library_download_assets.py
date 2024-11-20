def download_assets(assetsmanager_settings: Optional[dict]=None,
    configuration: Optional[Mapping[str, Union[Dict[str, Any],
    ModelConfiguration]]]=None, models: Optional[LibraryModelsType]=None,
    required_models: Optional[List[str]]=None) ->Tuple[Dict[str, Set[str]],
    Dict[str, AssetInfo]]:
    assetsmanager_settings = assetsmanager_settings or {}
    assets_manager = AssetsManager(**assetsmanager_settings)
    configuration = configure(models=models, configuration=configuration)
    models_assets = {}
    assets_info = {}
    required_models = required_models or [r for r in configuration]
    for model in required_models:
        models_assets[model] = list_assets(required_models=[model],
            configuration=configuration)
        for asset in models_assets[model]:
            if asset in assets_info:
                continue
            assets_info[asset] = AssetInfo(**assets_manager.fetch_asset(
                asset, return_info=True))
    return models_assets, assets_info
