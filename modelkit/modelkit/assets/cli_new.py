@assets_cli.command('new')
@click.argument('asset_path')
@click.argument('asset_spec')
@click.option('--storage-prefix', envvar='MODELKIT_STORAGE_PREFIX')
@click.option('--dry-run', is_flag=True)
def new(asset_path, asset_spec, storage_prefix, dry_run):
    """
    Create a new asset.

    Create a new asset ASSET_SPEC with ASSET_PATH file.

    Will fail if asset exists (in this case use `update`).

    ASSET_PATH is the path to the file. The file can be local or on GCS
    (starting with gs://)

    ASSET_SPEC is and asset specification of the form
    [asset_name] (Major/minor version information is ignored)

    NB: [asset_name] can contain `/` too.
    """
    new_(asset_path, asset_spec, storage_prefix, dry_run)
