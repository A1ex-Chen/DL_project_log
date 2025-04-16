@assets_cli.command('fetch')
@click.argument('asset')
@click.option('--download', is_flag=True)
def fetch_asset(asset, download):
    """Fetch an asset and download if necessary"""
    manager = AssetsManager()
    info = manager.fetch_asset(asset, return_info=True, force_download=download
        )
    console = Console()
    console.print(info)
