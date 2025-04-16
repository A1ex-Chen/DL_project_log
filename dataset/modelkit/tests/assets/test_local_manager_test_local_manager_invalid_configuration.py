def test_local_manager_invalid_configuration(working_dir):
    modelkit_storage_bucket = working_dir
    modelkit_storage_prefix = 'assets-prefix'
    modelkit_assets_dir = os.path.join(modelkit_storage_bucket,
        modelkit_storage_prefix)
    os.makedirs(modelkit_assets_dir)
    with pytest.raises(errors.StorageDriverError):
        AssetsManager(assets_dir=modelkit_assets_dir, storage_provider=
            StorageProvider(provider='local', prefix=
            modelkit_storage_prefix, bucket=modelkit_storage_bucket))
