@pytest.fixture(scope='function')
def local_assetsmanager(base_dir, working_dir):
    bucket_path = os.path.join(base_dir, 'local_driver', 'bucket')
    os.makedirs(bucket_path)
    mng = AssetsManager(assets_dir=working_dir, storage_provider=
        StorageProvider(provider='local', bucket=bucket_path))
    yield mng
    _delete_all_objects(mng)
