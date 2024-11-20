@skip_unless('ENABLE_GCS_TEST', 'True')
def test_download_object_or_prefix_cli(gcs_assetsmanager):
    original_asset_path = os.path.join(test_path, 'testdata', 'some_data.json')
    provider = gcs_assetsmanager.storage_provider
    object_dir = f'{provider.prefix}/category-test/some-data.ext'
    object_name = object_dir + '/1.0'
    provider.push(original_asset_path, 'category-test/some-data.ext', '1.0')
    with tempfile.TemporaryDirectory() as tmp_dir:
        asset_path = modelkit.assets.cli._download_object_or_prefix(provider
            .driver, object_name=object_name, destination_dir=tmp_dir)
        assert filecmp.cmp(original_asset_path, asset_path)
    with tempfile.TemporaryDirectory() as tmp_dir:
        asset_dir = modelkit.assets.cli._download_object_or_prefix(provider
            .driver, object_name=object_dir, destination_dir=tmp_dir)
        assert filecmp.cmp(original_asset_path, os.path.join(asset_dir, '1.0'))
    with tempfile.TemporaryDirectory() as tmp_dir:
        with pytest.raises(modelkit.assets.errors.ObjectDoesNotExistError):
            modelkit.assets.cli._download_object_or_prefix(provider.driver,
                object_name=object_name + 'file-not-found', destination_dir
                =tmp_dir)
        with pytest.raises(modelkit.assets.errors.ObjectDoesNotExistError):
            modelkit.assets.cli._download_object_or_prefix(provider.driver,
                object_name=f'{provider.prefix}/category-test',
                destination_dir=tmp_dir)
