def test_create_modelkit_app_no_model(monkeypatch):
    monkeypatch.delenv('MODELKIT_DEFAULT_PACKAGE', raising=False)
    with pytest.raises(ModelsNotFound):
        create_modelkit_app(models=None)
