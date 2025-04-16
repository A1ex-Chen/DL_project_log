@assets_cli.command('update')
@click.argument('asset_path')
@click.argument('asset_spec')
@click.option('--bump-major', is_flag=True, help=
    '[minor-major] Push a new major version (1.0, 2.0, etc.)')
@click.option('--storage-prefix', envvar='MODELKIT_STORAGE_PREFIX')
@click.option('--dry-run', is_flag=True)
def update(asset_path, asset_spec, storage_prefix, bump_major, dry_run):
    """
    Update an existing asset using versioning system
    set in MODELKIT_ASSETS_VERSIONING_SYSTEM (major/minor by default)

    Update an existing asset ASSET_SPEC with ASSET_PATH file.


    By default will upload a new minor version.

    ASSET_PATH is the path to the file. The file can be local remote (AWS or GCS)
    (starting with gs:// or s3://)

    ASSET_SPEC is and asset specification of the form
    [asset_name]:[version]

    Specific documentation depends on the choosen model
    """
    try:
        update_(asset_path, asset_spec, storage_prefix, bump_major, dry_run)
    except ObjectDoesNotExistError:
        print('Remote asset not found. Create it first using `new`')
        sys.exit(1)
