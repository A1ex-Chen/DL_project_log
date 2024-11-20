def modellibrary_auto_test(configuration=None, models=None, required_models
    =None, fixture_name='testing_model_library', test_name=
    'testing_model_library', necessary_fixtures=None, fixture_scope=
    'session', test_dir='.'):
    import pytest
    test_cases = []
    configurations = configure(models=models, configuration=configuration)
    for model_key, model_configuration in configurations.items():
        if issubclass(model_configuration.model_type, Model):
            if required_models and model_key not in required_models:
                continue
            for model_key, item, result, kwargs in model_configuration.model_type._iterate_test_cases(
                ):
                test_cases.append((model_key, item, result, kwargs))

    @pytest.mark.parametrize('model_key, item, result, kwargs', test_cases)
    def test_function(model_key, item, result, kwargs, request):
        lib = request.getfixturevalue(fixture_name)
        pred = lib.get(model_key)(item, **kwargs)
        if isinstance(result, JSONTestResult):
            ref = ReferenceJson(os.path.join(test_dir, os.path.dirname(
                result.fn)))
            if isinstance(pred, pydantic.BaseModel):
                pred = pred.model_dump()
            ref.assert_equal(os.path.basename(result.fn), pred)
        elif has_numpy and isinstance(result, np.ndarray):
            assert np.array_equal(pred, result), f'{pred} != {result}'
        else:
            if isinstance(pred, pydantic.BaseModel) and isinstance(result, dict
                ):
                pred = pred.model_dump()
            assert pred == result, f'{pred} != {result}'
    frame = inspect.currentframe().f_back
    frame.f_locals[test_name] = test_function
