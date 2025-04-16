def test_modelkit_settings_working(monkeypatch):


    class ServingSettings(ModelkitSettings):
        enable: bool = pydantic.Field(False, validation_alias=pydantic.
            AliasChoices('enable', 'SERVING_ENABLE'))
    assert ServingSettings().enable is False
    assert ServingSettings(enable=True).enable is True
    monkeypatch.setenv('SERVING_ENABLE', 'True')
    assert ServingSettings().enable is True
    assert ServingSettings(enable=False).enable is False
