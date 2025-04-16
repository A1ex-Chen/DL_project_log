@pytest.mark.parametrize('model_key, item, result, kwargs', test_cases)
def test_function(model_key, item, result, kwargs, request):
    lib = request.getfixturevalue(fixture_name)
    pred = lib.get(model_key)(item, **kwargs)
    if isinstance(result, JSONTestResult):
        ref = ReferenceJson(os.path.join(test_dir, os.path.dirname(result.fn)))
        if isinstance(pred, pydantic.BaseModel):
            pred = pred.model_dump()
        ref.assert_equal(os.path.basename(result.fn), pred)
    elif has_numpy and isinstance(result, np.ndarray):
        assert np.array_equal(pred, result), f'{pred} != {result}'
    else:
        if isinstance(pred, pydantic.BaseModel) and isinstance(result, dict):
            pred = pred.model_dump()
        assert pred == result, f'{pred} != {result}'
