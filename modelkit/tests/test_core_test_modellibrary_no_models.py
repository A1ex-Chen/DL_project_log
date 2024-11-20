def test_modellibrary_no_models(monkeypatch):
    monkeypatch.setenv('modelkit_MODELS', '')
    p = ModelLibrary(models=None)
    assert p.configuration == {}
    assert p.required_models == {}
    with pytest.raises(errors.ModelsNotFound):
        p.get('some_model')
