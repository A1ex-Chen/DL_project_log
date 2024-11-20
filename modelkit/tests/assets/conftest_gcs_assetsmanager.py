@pytest.fixture(scope='function')
def gcs_assetsmanager(request, working_dir):
    subprocess.call(['docker', 'rm', '-f', 'modelkit-storage-gcs-tests'],
        stderr=subprocess.DEVNULL)
    minio_proc = subprocess.Popen(['docker', 'run', '-p', '4443:4443',
        '--name', 'modelkit-storage-gcs-tests', 'fsouza/fake-gcs-server'])

    def finalize():
        subprocess.call(['docker', 'stop', 'modelkit-storage-gcs-tests'])
        minio_proc.terminate()
        minio_proc.wait()
    request.addfinalizer(finalize)
    storage_provider = StorageProvider(prefix='test-prefix', provider='gcs',
        bucket='test-bucket', client=_get_mock_gcs_client())
    storage_provider.driver.client.create_bucket('test-bucket')
    mng = AssetsManager(assets_dir=working_dir, storage_provider=
        storage_provider)
    yield mng
