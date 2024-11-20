def fetch_asset(self, spec: Union[AssetSpec, str], return_info=False,
    force_download: typing.Optional[bool]=None):
    if isinstance(spec, str):
        spec = cast(AssetSpec, AssetSpec.from_string(spec))
    if force_download is None and self.storage_provider:
        force_download = self.storage_provider.force_download
    logger.info('Fetching asset...', name=spec.name, version=spec.version,
        return_info=return_info, force_download=force_download)
    asset_info = self._fetch_asset(spec, _force_download=force_download)
    path = asset_info['path']
    if not os.path.exists(path):
        logger.error(
            'An unknown error occured when fetching asset.The path does not exist.'
            , path=path, spec=spec)
        raise AssetFetchError(
            f'An unknown error occured when fetching asset {spec}.The path {path} does not exist.'
            )
    logger.info('Fetched asset', name=spec.name, version=spec.version,
        from_cache=asset_info.get('from_cache'))
    if not return_info:
        return path
    return asset_info
