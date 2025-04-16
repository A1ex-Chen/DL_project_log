def test_method__get_all_subs():
    model_library = ModelLibrary(models=[ModelA, ModelB, ModelC, ModelD,
        Pipeline])
    pipeline = model_library.get('pipeline')
    item = {'abc': 123}
    profiler = SimpleProfiler(pipeline)
    _ = pipeline.predict(item)
    assert profiler._get_all_subs('pipeline') == {'model_a', 'model_b',
        'model_d', 'model_c'}
    assert profiler._get_all_subs('model_a') == set()
    assert profiler._get_all_subs('model_b') == {'model_a'}
    assert profiler._get_all_subs('model_c') == set()
    assert profiler._get_all_subs('model_d') == {'model_a'}
