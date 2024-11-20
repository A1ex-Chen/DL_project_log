def update_(asset_path, asset_spec, storage_prefix, bump_major, dry_run):
    _check_asset_file_number(asset_path)
    destination_provider = StorageProvider(prefix=storage_prefix)
    print('Destination assets provider:')
    print(f' - storage driver = `{destination_provider.driver}`')
    print(f' - driver bucket = `{destination_provider.driver.bucket}`')
    print(f' - prefix = `{storage_prefix}`')
    print(f'Current asset: `{asset_spec}`')
    versioning_system = os.environ.get('MODELKIT_ASSETS_VERSIONING_SYSTEM',
        'major_minor')
    spec = AssetSpec.from_string(asset_spec, versioning=versioning_system)
    print(f' - versioning system = `{versioning_system}` ')
    print(f' - name = `{spec.name}`')
    print(f' - version = `{spec.version}`')
    version_list = destination_provider.get_versions_info(spec.name)
    update_params = spec.versioning.get_update_cli_params(version=spec.
        version, version_list=version_list, bump_major=bump_major)
    print(update_params['display'])
    new_version = spec.versioning.increment_version(spec.sort_versions(
        version_list), update_params['params'])
    print(f'Push a new asset version `{new_version}` for `{spec.name}`?')
    response = click.prompt('[y/N]')
    if response == 'y':
        with tempfile.TemporaryDirectory() as tmp_dir:
            if not os.path.exists(asset_path):
                parsed_path = parse_remote_url(asset_path)
                if parsed_path['storage_prefix'] == 'gs':
                    if not has_gcs:
                        raise DriverNotInstalledError(
                            'GCS driver not installed, install modelkit[assets-gcs]'
                            )
                    driver_settings = GCSStorageDriverSettings(bucket=
                        parsed_path['bucket_name'])
                    driver = GCSStorageDriver(driver_settings)
                elif parsed_path['storage_prefix'] == 's3':
                    if not has_s3:
                        raise DriverNotInstalledError(
                            'S3 driver not installed, install modelkit[assets-s3]'
                            )
                    driver_settings = S3StorageDriverSettings(bucket=
                        parsed_path['bucket_name'])
                    driver = S3StorageDriver(driver_settings)
                else:
                    raise ValueError(
                        f"Unmanaged storage prefix `{parsed_path['storage_prefix']}`"
                        )
                asset_path = _download_object_or_prefix(driver, object_name
                    =parsed_path['object_name'], destination_dir=tmp_dir)
            destination_provider.update(asset_path, name=spec.name, version
                =new_version, dry_run=dry_run)
        return new_version
    print('Aborting.')
