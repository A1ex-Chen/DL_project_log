def modellibrary_fixture(settings=None, assetsmanager_settings=None,
    configuration=None, models=None, required_models=None, fixture_name=
    'testing_model_library', necessary_fixtures=None, fixture_scope='session'):
    import pytest

    @pytest.fixture(name=fixture_name, scope=fixture_scope)
    def fixture_function(request):
        if necessary_fixtures:
            for fixture_name in necessary_fixtures:
                request.getfixturevalue(fixture_name)
        return ModelLibrary(settings=settings, assetsmanager_settings=
            assetsmanager_settings, configuration=configuration, models=
            models, required_models=required_models)
    frame = inspect.currentframe().f_back
    frame.f_locals[fixture_name] = fixture_function
