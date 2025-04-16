@pytest.mark.parametrize(
    'required_models_env_var, models, required_models, n_endpoints', [(None,
    [SomeSimpleValidatedModel, SomeComplexValidatedModel], [], 4), (
    'some_model:some_complex_model', [SomeSimpleValidatedModel,
    SomeComplexValidatedModel], [], 8), ('some_complex_model', [
    SomeSimpleValidatedModel, SomeComplexValidatedModel], [], 6), (None, [
    SomeSimpleValidatedModel, SomeComplexValidatedModel], ['some_model',
    'some_complex_model'], 8), (None, [SomeSimpleValidatedModel,
    SomeComplexValidatedModel], ['some_model'], 6)])
def test_create_modelkit_app(required_models_env_var, models,
    required_models, n_endpoints, monkeypatch):
    if required_models_env_var:
        monkeypatch.setenv('MODELKIT_REQUIRED_MODELS', required_models_env_var)
    app = create_modelkit_app(models=models, required_models=required_models)
    assert len([route.path for route in app.routes]) == n_endpoints
