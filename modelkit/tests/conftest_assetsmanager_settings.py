@pytest.fixture
def assetsmanager_settings(working_dir):
    yield {'storage_provider': StorageProvider(prefix='assets-prefix',
        provider='local', bucket=os.path.join(TEST_DIR, 'testdata',
        'test-bucket')), 'assets_dir': working_dir}
