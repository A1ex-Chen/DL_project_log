@click.command()
@click.argument('assets_dir')
@click.argument('driver_path')
@click.argument('asset_name')
def download_asset(assets_dir, driver_path, asset_name):
    """
    Download the asset
    """
    am = AssetsManager(assets_dir=assets_dir, storage_provider=
        StorageProvider(provider='local', bucket=driver_path, prefix='prefix'))
    asset_dict = am.fetch_asset(asset_name, return_info=True)
    if asset_dict['from_cache']:
        print('__ok_from_cache__')
    else:
        print('__ok_not_from_cache__')
