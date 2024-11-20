def test_assetsmanager_default_assets_dir():
    manager = AssetsManager()
    assert manager.assets_dir == os.getcwd()
    assert manager.storage_provider is None
