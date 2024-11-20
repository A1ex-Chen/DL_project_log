@pytest.mark.parametrize('Settings', [StorageDriverSettings,
    GCSStorageDriverSettings, AzureStorageDriverSettings,
    S3StorageDriverSettings, LocalStorageDriverSettings])
def test_storage_driver_settings(Settings, monkeypatch):
    monkeypatch.setenv('MODELKIT_STORAGE_BUCKET', 'foo')
    assert Settings().bucket == 'foo'
    assert Settings(bucket='bar').bucket == 'bar'
    monkeypatch.delenv('MODELKIT_STORAGE_BUCKET')
    assert Settings(bucket='bar').bucket == 'bar'
    with pytest.raises(pydantic.ValidationError):
        _ = Settings()
