def new_(asset_path, asset_spec, storage_prefix, dry_run):
    _check_asset_file_number(asset_path)
    destination_provider = StorageProvider(prefix=storage_prefix)
    print('Destination assets provider:')
    print(f' - storage driver = `{destination_provider.driver}`')
    print(f' - driver bucket = `{destination_provider.driver.bucket}`')
    print(f' - prefix = `{storage_prefix}`')
    print(f'Current asset: `{asset_spec}`')
    spec = AssetSpec.from_string(asset_spec)
    version = spec.versioning.get_initial_version()
    print(f' - name = `{spec.name}`')
    print(f'Push a new asset `{spec.name}` with version `{version}`?')
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
            destination_provider.new(asset_path, spec.name, version, dry_run)
        return version
    print('Aborting.')
