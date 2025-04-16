def _fetch_local_version(asset_name: str, local_name: str) ->Dict[str, str]:
    if os.path.exists(local_name):
        logger.debug('Asset is a valid local path relative to ASSETS_DIR',
            local_name=local_name)
        return {'path': local_name}
    path = os.path.join(os.getcwd(), *asset_name.split('/'))
    if os.path.exists(path):
        logger.debug('Asset is a valid relative local path', local_name=path)
        return {'path': path}
    if os.path.exists(asset_name):
        logger.debug('Asset is a valid absolute local path', local_name=path)
        return {'path': asset_name}
    raise errors.AssetDoesNotExistError(asset_name)
