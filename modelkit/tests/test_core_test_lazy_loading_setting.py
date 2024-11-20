def test_lazy_loading_setting(monkeypatch):
    monkeypatch.delenv('MODELKIT_LAZY_LOADING', raising=False)
    settings = LibrarySettings()
    assert not settings.lazy_loading
    monkeypatch.setenv('MODELKIT_LAZY_LOADING', 'True')
    settings = LibrarySettings()
    assert settings.lazy_loading
