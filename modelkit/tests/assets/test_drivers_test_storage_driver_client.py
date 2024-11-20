def test_storage_driver_client(monkeypatch):


    class MockedDriver(StorageDriver):

        @staticmethod
        def build_client(_):
            return {'built': True, 'passed': False}

        def delete_object(self, object_name: str):
            ...

        def download_object(self, object_name: str, destination_path: str):
            ...

        def exists(self, object_name: str) ->bool:
            ...

        def upload_object(self, file_path: str, object_name: str):
            ...

        def get_object_uri(self, object_name: str, sub_part: Optional[str]=None
            ):
            ...

        def iterate_objects(self, prefix: Optional[str]=None):
            ...
    settings = StorageDriverSettings(bucket='bucket')
    driver = MockedDriver(settings, client={'built': False, 'passed': True})
    assert settings.lazy_driver is False
    assert driver._client is not None
    assert driver._client == driver.client == {'built': False, 'passed': True}
    driver = MockedDriver(settings)
    assert settings.lazy_driver is False
    assert driver._client is not None
    assert driver._client == driver.client == {'built': True, 'passed': False}
    monkeypatch.setenv('MODELKIT_LAZY_DRIVER', True)
    settings = StorageDriverSettings(bucket='bucket')
    driver = MockedDriver(settings)
    assert settings.lazy_driver is True
    assert driver._client is None
    assert driver.client == {'built': True, 'passed': False}
    assert driver._client is None
    driver = MockedDriver(settings, client={'built': False, 'passed': True})
    assert driver._client is not None
    assert driver._client == driver.client == {'built': False, 'passed': True}
