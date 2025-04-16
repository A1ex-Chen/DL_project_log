def test_method__get_current_sub_calls():
    model_library = ModelLibrary(models=[ModelA, ModelB, ModelC, ModelD,
        Pipeline])
    pipeline = model_library.get('pipeline')
    item = {'abc': 123}
    profiler = SimpleProfiler(pipeline)
    _ = pipeline.predict(item)
    assert profiler._get_current_sub_calls('pipeline') == {'model_a': 2,
        'model_b': 1, 'model_d': 1, 'model_c': 1}
    assert profiler._get_current_sub_calls('model_a') == {}
    assert profiler._get_current_sub_calls('model_b') == {'model_a': 2}
    assert profiler._get_current_sub_calls('model_c') == {}
    assert profiler._get_current_sub_calls('model_d') == {'model_a': 2}
