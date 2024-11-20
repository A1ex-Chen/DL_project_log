@pytest.fixture(name=fixture_name, scope=fixture_scope)
def fixture_function(request):
    if necessary_fixtures:
        for fixture_name in necessary_fixtures:
            request.getfixturevalue(fixture_name)
    return ModelLibrary(settings=settings, assetsmanager_settings=
        assetsmanager_settings, configuration=configuration, models=models,
        required_models=required_models)
